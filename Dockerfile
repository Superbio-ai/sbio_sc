FROM 221497708189.dkr.ecr.us-west-2.amazonaws.com/ml_resources:pytorch_gpu_ecs_470

RUN apt-get update
RUN apt-get install vim -y
RUN apt-get install build-essential procps curl file git -y
RUN apt-get update && apt-get install -y --no-install-recommends gcc

RUN pip install --upgrade pip
RUN mkdir /app
COPY requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN pip install torchtext
RUN pip install pytorch-lightning

# ##ENV
ARG JOB_ID
ENV JOB_ID=$JOB_ID
ARG AWS_ACCESS_KEY_ID
ENV AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY
ENV AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
ARG AWS_REGION='us-west-2'
ENV AWS_REGION=$AWS_REGION
ARG AWS_DATASET_BUCKET
ENV AWS_DATASET_BUCKET=$AWS_DATASET_BUCKET
ARG APP_USER
ENV APP_USER=$APP_USER
ARG APP_USER_PASSWORD
ENV APP_USER_PASSWORD=$APP_USER_PASSWORD
ARG SBIO_API_URL
ENV SBIO_API_URL=$SBIO_API_URL

#CMD python3 /app/app_runner.py

ENTRYPOINT python /app/app_runner.py ${JOB_ID}