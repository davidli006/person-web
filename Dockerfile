FROM python

WORKDIR /app

COPY ./person-web /app

RUN python -m pip install --upgrade pip  -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com
RUN pip install -r ./requirments.txt -i http://pypi.douban.com/simple/ --trusted-host pypi.douban.com

RUN echo 'deb http://mirrors.163.com/debian/ stretch main non-free contrib' > /etc/apt/sources.list
RUN echo 'deb http://mirrors.163.com/debian/ stretch-updates main non-free contrib' >> /etc/apt/sources.list
RUN echo 'deb http://mirrors.163.com/debian-security/ stretch/updates main non-free contrib' >> /etc/apt/sources.list
RUN apt update
RUN apt install -y iproute2


ENTRYPOINT  ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "main:app"]
