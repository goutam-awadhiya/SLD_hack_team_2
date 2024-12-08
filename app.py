from flask import Flask, render_template, request, jsonify, json
import cv2
import boto3
import os
import json
import base64
from song_list import emotion_to_songs
app = Flask(__name__)
os.getenv("AWS_SECRET_ACCESS_KEY")
print(os.getenv("AWS_SECRET_ACCESS_KEY"))
# Directory to save captured faces
UPLOAD_FOLDER = 'captured_faces'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
# os.environ['AWS_DEFAULT_REGION']= <>
# os.environ['AWS_ACCESS_KEY_ID']= <>
# os.environ['AWS_SECRET_ACCESS_KEY']= <>
# os.environ['AWS_SESSION_TOKEN'] = <>
rekognition_client = boto3.client(
    'rekognition',
    # region_name='us-west-2',  # e.g., 'us-east-1'
    # aws_access_key_id="AKIAR24PSXYL3PLCUZTR",
    # aws_secret_access_key="2FJBiAKlpeGGL0vZVRzF1LiMDav007oj+brD6L3/"
)
bedrock_client = boto3.client(
    "bedrock-runtime"
    # region_name='us-west-2',  # e.g., 'us-east-1'
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


@app.route('/chat', methods=['POST'])
def chat():
    print("line 72")
    data = request.json
    # user_message = data.get("message", "")
    emotion = data.get("emotion", "neutral").lower()
    print(emotion)

    print("line 77")
    # Prepare chatbot prompt
    # prompt = (
    #     f"You are a bollywood music expert and understand what music can influence {emotion} mood of a human being." 
    #     f"Your name is Mood Melody."
    #     f"Please provide 5 relevant songs as youtube links and empathetic advice."
    #     f"Each response should include:"
    #     f"1. An empathetic acknowledgment of the {emotion},"
    #     f"2. 5 song recommendations with valid YouTube links,"
    #     f"3. A follow-up question to check on the user's mood."
    #     # f"User: {user_message}\nBot:"
    # )
    # prompt = (
    #     f"I amd feeling {emotion}, please be empathetic if the emotion is negative and suggest me some song names. Do not ask questions."
    # )
    prompt = (
        f"You are a music recommendation chatbot named Mood Melody. The user feels {emotion}. "
        f"Generate a friendly response and suggest songs matching the mood with YouTube links.\n\n"
        # f"User: {user_message}\nBot:"
    )
    print("line 84")
    try:
        # Call Bedrock to generate chatbot response
        # modelId="amazon.titan-text-express-v1",
        response = bedrock_client.invoke_model(
            modelId="anthropic.claude-3-5-sonnet-20241022-v2:0",
            contentType="application/json",
            body = json.dumps({"max_tokens": 256,"messages": [{"role": "user", "content": f"I'm feeling {emotion}, please acknowledge the {emotion} emotion and suggest me 5 bollywood youtube songs and don't ask further questions. Feel free to add emoji and youtube links for recommended songs"}],"anthropic_version": "bedrock-2023-05-31"}))
            # body={"inputText": prompt}
            # body=f"{{\"inputText\": \"{prompt}\"}}"
            # body=json.dumps({"inputText": prompt})
            # body={"inputText": prompt,
            #       "parameters": {"max_tokens": 300, "temperature": 0.7}}
        # )
        print("received response")
        response_body = json.loads(response.get("body").read())
        print(response_body)
        print(response_body.get("content")) 
        # bot_reply = response['body'].read().decode('utf-8')
#         bot_reply_parsed = json.loads(response["bot_reply"])
# # Update the response dictionary
#         response["bot_reply"] = bot_reply_parsed
#         # Output the formatted dictionary
#         print(json.dumps(response, indent=4))

    except Exception as e:
        bot_reply = f"Sorry, I couldn't process your request. Error: {str(e)}"

    # print(bot_reply)
    # Fetch songs based on detected emotion
    # songs = emotion_to_songs.get(emotion, [{"title": "No suggestions available", "url": "#"}])
    # return jsonify({"bot_reply": bot_reply, "songs": songs})
    return jsonify({"bot_reply": response_body.get("content")})

if __name__ == '__main__':
    # chat()
    app.run(debug=True)
