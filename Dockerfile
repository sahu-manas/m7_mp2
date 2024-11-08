# pull python base image
FROM python:3.10

ADD requirements.txt requirements.txt


# update pip
RUN pip install --upgrade pip

# install dependencies
RUN pip install -r requirements.txt

# copy application files
COPY app/. app/.

# expose port for application
EXPOSE 8001

# start fastapi application
CMD ["python", "app/main.py"]