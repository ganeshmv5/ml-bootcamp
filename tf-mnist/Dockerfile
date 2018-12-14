#This container contains your model and any helper scripts specific to your model.
#FROM tensorflow/tensorflow:1.7.0
FROM tensorflow/tensorflow

ADD requirements.txt /opt/requirements.txt

RUN apt-get update -y
RUN apt-get install -y python-pip python-dev build-essential python-setuptools
RUN pip install --upgrade pip
RUN pip install -r /opt/requirements.txt

ADD data_csv_filtered1.csv /opt/data_csv_filtered1.csv
ADD mnist_model.py /opt/mnist_model.py

RUN chmod +x /opt/mnist_model.py

ENTRYPOINT ["/usr/bin/python"]
CMD ["/opt/mnist_model.py"]
