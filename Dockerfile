FROM python:3.11-slim

# Working Directory
WORKDIR /app

# Copy source code to working directory
COPY . /app

LABEL org.opencontainers.image.source https://github.com/Medkallel/Gift-Recommendation-Chatbot

# Install packages from requirements.txt

RUN pip install --no-cache-dir --upgrade pip &&\
    pip install --no-cache-dir --trusted-host pypi.python.org -r requirements.txt

CMD python -m streamlit run /app/src/Gift_Recommendation_Bot.py