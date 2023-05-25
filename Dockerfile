FROM waggle/plugin-base:0.1.0

# Update apt packages
RUN apt update
RUN apt upgrade -y

RUN apt-get update \
    && apt-get install -y \
       vim cmake libsm6 libxext6 libxrender-dev protobuf-compiler \
    && rm -r /var/lib/apt/lists/*


RUN apt update
# Install python 3.7
RUN apt install software-properties-common -y
RUN add-apt-repository ppa:deadsnakes/ppa
RUN apt install python3.7 -y

# Make python 3.7 the default
RUN echo "alias python=python3.7" >> ~/.bashrc
RUN export PATH=${PATH}:/usr/bin/python3.7
RUN /bin/bash -c "source ~/.bashrc"

# Install pip
RUN apt install python3-pip -y
RUN python3 -m pip install --upgrade pip
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python3", "plugin.py"]
