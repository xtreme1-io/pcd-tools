import io
import requests
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from alg_service import *
from pcd_tools import *
from pcd_tools.metrics.map import build_metrics


D_RESOLUTION = 1000
D_NUM_STD = 3
D_COLORS = (0x6055C6, 0x378CDF, 0x1CC7C1, 0x33EC83, 0x7FF55B)
D_Z_RANGE = None


def _build_item_error(code: int, message: str):
    return {
        "code": code,
        "message": message
    }

class AppHandler(BaseApiHandler):
    # override
    def post(self):
        args = self.args
        data = self.get_field(args, key='data', type_=list, check_empty=True)
        task_type = self.get_field(args, key='type', type_=int)
        if task_type < 1 or task_type > 3:
            raise ValueError(f"invalid 'type' value: {task_type}")

        # renderParam
        renderParam = args.get("renderParam", {})
        colors = renderParam.get('colors', D_COLORS)
        zRange = renderParam.get('zRange', D_Z_RANGE)
        width = renderParam.get('width', D_RESOLUTION)
        height = renderParam.get('height', D_RESOLUTION)
        num_std = renderParam.get('numStd', D_NUM_STD)
        
        pcd2image = PC2Image(colors, zRange, (width, height), num_std)

        # convertParam
        convertParam = args.get("convertParam", {})
        extra_fields = convertParam.get('extraFields', [])

        # process
        num_samples = len(data)
        if num_samples > 1:
            max_workers =  min(8, num_samples)
            logging.info(f"using ThreadPoolExecutor with {max_workers} workers")

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                results = list(executor.map(
                    partial(self.process_data, task_type=task_type, pcd2image=pcd2image, extra_fields=extra_fields), 
                    data))
        else:
            results = [self.process_data(item, task_type, pcd2image, extra_fields) for item in data]
        self.return_ok(results)

    def process_data(self, item, task_type, pcd2image, extra_fields):
        if not isinstance(item, dict):
            return _build_item_error(1, "data item must be a dictionary")

        pc_file = item.get("pointCloudFile", None)
        if pc_file is None:
            return _build_item_error(1, "missing 'pointCloudFile'")

        if not isinstance(pc_file, str) or not pc_file.startswith("http"):
            return _build_item_error(1, "invalid 'pointCloudFile'")

        result = {
            "code": 0,
            "message": "success"
        }
        try:
            t = Timing()
            logging.info(f"{'-'*10} {pc_file} {'-'*10}")

            r = requests.get(pc_file, allow_redirects=True)
            t.log_interval(f"DOWNLOAD pcd({len(r.content)/1024/1024:.1f}MB)")

            pcd_path = io.BytesIO(r.content)
            pc = PointCloud(pcd_path)
            if pc.invalid_points > 0:
                logging.info(f"\t{pc.invalid_points} invalid points removed")
            t.log_interval(f'LOAD pcd')

            # bev image
            if task_type != 2:
                upload_image_path = item.get("uploadImagePath", None)
                if upload_image_path is None:
                    return _build_item_error(1, "missing 'uploadImagePath'")

                image, (left, top, right, bottom) = pcd2image.bev_from_pc(pc.normalized_numpy())
                with io.BytesIO() as output:
                    image.save(output, format="PNG")
                    contents = output.getvalue()
                result.update({
                    'pointCloudRange': {
                        'left': left,
                        'top': top,
                        'right': right,
                        'bottom': bottom
                    },
                    'imageSize': len(contents)
                })
                t.log_interval("GEN image")

                # upload image
                r = requests.put(upload_image_path, data=contents, headers={"Content-Type":"application/binary"})
                if r.status_code != 200:
                    msg = f"upload image failed({r.status_code}): '{upload_image_path}'\n{r.text}"
                    logging.warn(msg)
                    return _build_item_error(2, msg)

                t.log_interval(f"UPLOAD image({len(contents)/1024:.0f}KB)")

            # normalize point cloud
            if task_type != 3:
                upload_binary_pcd_path = item.get("uploadBinaryPcdPath", None)
                if upload_binary_pcd_path is None:
                    return _build_item_error(1, "missing 'uploadBinaryPcdPath'")

                npc = pc.normalized_pc(extra_fields=extra_fields)
                with io.BytesIO() as output:
                    pc.save_pcd(npc, output)
                    contents = output.getvalue()
                result['binaryPcdSize'] = len(contents)
                # t.log_interval("GEN pcd")

                # upload point cloud
                r = requests.put(upload_binary_pcd_path, data=contents, headers={"Content-Type":"application/binary"})
                if r.status_code != 200:
                    msg = f"upload pcd failed({r.status_code}): '{upload_binary_pcd_path}'"
                    logging.warn(msg)
                    return _build_item_error(2, msg)

                t.log_interval(f"UPLOAD pcd({len(contents)/1024/1024:.1f}MB)\n{r.text}")

            logging.info(f"--- pcd info: {pc.code}, {pc.fields}, {len(pc.data):,} points")

        except Exception as e:
            logging.exception(e)
            return _build_item_error(2, str(e))

        return result

class EvaluateHandler(BaseApiHandler):
    # override
    def post(self):
        args = self.args
        gtUrl = self.get_field(args, key='groundTruthResultFileUrl', type_=str, check_empty=True)
        modelUrl = self.get_field(args, key='modelRunResultFileUrl', type_=str, check_empty=True)

        try:
            t = Timing()
            targets, l = self.load_objects_from_file(gtUrl)
            t.log_interval(f"DOWNLOAD groundTruthResultFileUrl({l/1024:.1f}KB)")

            preds, l = self.load_objects_from_file(modelUrl)
            t.log_interval(f"DOWNLOAD modelRunResultFileUrl({l/1024:.1f}KB)")

            metrics = build_metrics(preds, targets)
            logging.info(f"---metrics: {json.dumps(metrics)}")

            ret_metrics = [
                {
                    "name": name,
                    "value": round(value, 4),
                    "description": f"{'mean ' if name.startswith('mAP') else ''}average precision for {name.split('_')[-1]}"
                }
                for name, value in metrics.items()
                if name.startswith('mAP') or name.startswith('AP')
            ]
            if len(metrics) > 2:
                ret_metrics.extend([
                    {
                        "name": f"AP-{type}-{label}",
                        "value": round(m[type]['AP'], 4),
                        "description": f"average precision for {label}"
                    }
                    for label, m in metrics.items()
                    for type in ['bev', '3d']
                    if not label.startswith('mAP') and not label.startswith('AP')
                ]) 

            self.return_ok({
                "metrics": ret_metrics,
            })

        except json.decoder.JSONDecodeError as e:
            logging.exception(e)
            return self.return_error("parse file failed, make sure each line is a json string")
        
        except KeyError as e:
            logging.exception(e)
            return self.return_error(f"missing key: {e}")

        except Exception as e:
            logging.exception(e)
            return _build_item_error(2, str(e))


    @staticmethod
    def load_objects_from_file(file):
        r = requests.get(file, allow_redirects=True)
        f = io.BytesIO(r.content)
        objects = [json.loads(line)['objects'] for line in f if line.strip()]
        return objects, len(r.content)

def main():
    parser = ArgumentParser()
    args = parse_args(parser)

    start_service([
            (r'/pointcloud/convert_render', AppHandler),
            (r'/pointCloud/resultEvaluate', EvaluateHandler),
        ],
        args)


if __name__ == '__main__':
    main()
