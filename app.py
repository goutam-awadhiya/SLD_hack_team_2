from flask import Flask, render_template, request, jsonify
import cv2
import boto3
import os
import base64
from song_list import emotion_to_songs
app = Flask(__name__)


# Directory to save captured faces
UPLOAD_FOLDER = 'captured_faces'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Configure AWS Rekognition
rekognition_client = boto3.client(
    'rekognition',
    region_name='us-west-2',  # e.g., 'us-east-1'
    aws_access_key_id="AKIAR24PSXYL3PLCUZTR",
    aws_secret_access_key="2FJBiAKlpeGGL0vZVRzF1LiMDav007oj+brD6L3/"
)
bedrock_client = boto3.client(
    "bedrock-runtime",
    region_name="us-west-2",  # Replace with your AWS region
    aws_access_key_id="your-access-key-id",
    aws_secret_access_key="your-secret-access-key"
)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/discover')
def discover():
    return render_template('index.html')



@app.route('/capture', methods=['POST'])
def capture():
    data = request.json
    img_data = data.get('image')

    if not img_data:
        return jsonify({"error": "No image data"}), 400

    # Decode base64 image and save it
    img_bytes = base64.b64decode(img_data.split(",")[1])
    img_path = os.path.join(UPLOAD_FOLDER, 'captured_face.jpg')

    with open(img_path, 'wb') as img_file:
        img_file.write(img_bytes)

    # Send image to Amazon Rekognition
    with open(img_path, 'rb') as image_file:
        response = rekognition_client.detect_faces(
            Image={'Bytes': image_file.read()},
            Attributes=['ALL']
        )

    # Parse emotion data
    emotions = response['FaceDetails'][0]['Emotions'] if response['FaceDetails'] else []
    dominant_emotion = max(emotions, key=lambda e: e['Confidence'], default={"Type": "NEUTRAL"})["Type"]

    # Suggest songs based on dominant emotion
    songs = emotion_to_songs.get(dominant_emotion.upper(), [{"title": "No suggestions available", "url": "#"}])

    return jsonify({"emotions": emotions, "songs": songs})


app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    # user_message = data.get("message", "")
    emotion = data.get("emotion", "neutral").lower()

    # Prepare chatbot prompt
    prompt = (
        f"You are a music recommendation chatbot named Mood Melody. The user feels {emotion}. "
        f"Generate a friendly response and suggest songs matching the mood with YouTube links.\n\n"
        # f"User: {user_message}\nBot:"
    )
    print("line 82")
    try:
        # Call Bedrock to generate chatbot response
        response = bedrock_client.invoke_model(
            modelId="amazon.titan-text",
            contentType="application/json",
            body={"inputText": prompt}
        )
        print("received response")
        bot_reply = response['body'].read().decode('utf-8')

    except Exception as e:
        bot_reply = f"Sorry, I couldn't process your request. Error: {str(e)}"

    # Fetch songs based on detected emotion
    songs = emotion_to_songs.get(emotion, [{"title": "No suggestions available", "url": "#"}])
    return jsonify({"bot_reply": bot_reply, "songs": songs})

if __name__ == '__main__':
    # chat()
    app.run(debug=True)
