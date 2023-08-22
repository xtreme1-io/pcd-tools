FROM ubuntu:20.04

RUN apt-get clean && \
    apt-get update && \
    apt-get install -y python3.8 python3-pip && \
    ln -svf /usr/bin/python3 /usr/bin/python

WORKDIR /app
COPY . ./
RUN pip install -e .

WORKDIR /app/pcd_tools
EXPOSE 5000
ENTRYPOINT ["python", "app.py"]
