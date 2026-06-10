# Humanoid Head

Humanoid Head is a modular robotics project focused on building an interactive humanoid robot head. The project combines computer vision, speech processing, natural language understanding, local language models, text-to-speech synthesis, servo-driven facial animation, and 3D-printed mechanical design.

The system is designed to recognize people, detect visible emotions, understand spoken questions, generate natural voice responses, and express emotions through physical facial movement.

## Project Overview

The robot head is built as a distributed system using two main computing devices:

- Raspberry Pi 5, responsible for vision processing, camera input, face recognition, emotion recognition, and facial actuation.
- NVIDIA Jetson Orin Nano, responsible for speech recognition, natural language processing, local language model inference, and speech synthesis.

The project is divided into three main modules:

- `vision-module`
- `speech-module`
- `hardware-module`

Each module was developed separately and later integrated into the final project.

## System Architecture

```text
User
 │
 │  voice, face, emotion
 ▼
Camera and Microphone
 │
 ├── Raspberry Pi 5
 │   ├── Face detection
 │   ├── Face recognition
 │   ├── Emotion recognition
 │   ├── Servo control
 │   └── Facial expression control
 │
 └── NVIDIA Jetson Orin Nano
     ├── Speech-to-text
     ├── Intent detection
     ├── Entity extraction
     ├── Local LLM response generation
     ├── Text-to-speech
     └── Conversation logic
```

The Raspberry Pi and Jetson communicate over the local network. The Raspberry Pi provides the speech pipeline with visual context, such as whether a person is visible, who the person is, and what emotion was detected. The Jetson generates spoken responses and sends mouth movement information back to the Raspberry Pi so that the robot jaw can move in sync with speech.

## How the Project Works

The project works as a real-time interaction loop between the user, the vision system, the speech system, and the physical robot head.

### 1. User Detection

The camera captures a live video stream. The vision module processes frames from the camera and searches for a visible face.

When a face is detected, the system determines:

- whether a person is currently in front of the robot,
- whether the person is already known,
- what emotion is visible on the face.

This information is sent to the speech module so that the robot can respond in a more context-aware way.

### 2. Face Recognition

The vision module can identify known users using stored face data. If the user is recognized, their identity is passed to the speech module.

This allows the robot to personalize its responses, for example by addressing the user by name or remembering context during an active interaction.

If the user is unknown, the system can treat the interaction as a new user session.

### 3. Emotion Recognition

The vision module classifies the user’s facial expression into emotional states such as:

- happy,
- sad,
- angry,
- fear,
- disgust,
- surprise,
- neutral.

The detected emotion is passed to the speech module and can also influence the robot’s physical facial expression.

The physical head supports servo-driven expression changes, allowing the robot to imitate or react to emotional states.

### 4. Speech Recognition

The speech module listens for the user’s spoken question or command.

Speech is converted to text using a speech-to-text pipeline based on Whisper or Faster Whisper. The transcribed text is then passed into the natural language processing pipeline.

The system is designed for spoken interaction, so the conversation flow assumes that the user is standing in front of the robot and speaking naturally.

### 5. Intent Detection

After transcription, the system determines what kind of request the user made.

The speech module can recognize different categories of intent, including:

- general conversation,
- weather questions,
- questions about rooms,
- questions about teachers,
- questions about auditoriums or special locations,
- project-specific or assistant-style questions.

Intent detection allows the system to choose whether it should answer using a local database, a rule-based lookup, or a local language model.

### 6. Entity Extraction

For location-related questions, the system extracts important entities from the user’s sentence.

Examples of extracted entities include:

- room numbers,
- building identifiers,
- teacher names,
- auditorium names,
- special places such as libraries, cloakrooms, or cafeterias.

The project uses GLiNER and supporting utility logic to detect and normalize these entities before searching the local data files.

### 7. Local Knowledge Lookup

For structured questions, the system searches local data files instead of relying only on a language model.

The speech module includes logic for:

- finding room directions,
- finding teacher room assignments,
- searching for special rooms,
- handling auditorium-related questions,
- retrieving weather information.

This gives the robot more reliable answers for known project-specific information.

### 8. Local LLM Response Generation

For general conversation, the system uses a local language model through Ollama and LangChain.

The language model receives context such as:

- the user’s question,
- recent conversation history,
- recognized user identity,
- detected emotion.

This allows the robot to generate more natural and context-aware spoken responses.

The prompt is designed specifically for a physical voice assistant. It avoids text-chat behavior and encourages natural spoken answers suitable for text-to-speech output.

### 9. Text-to-Speech

After generating a response, the speech module converts the text into audio using Piper TTS.

The generated audio is played through the speaker connected to the Jetson.

At the same time, the speech module estimates the audio intensity and sends mouth movement values to the Raspberry Pi. This allows the jaw servo to move in sync with the spoken response.

### 10. Facial Movement and Servo Control

The Raspberry Pi controls the robot’s servos through a servo controller.

The hardware system includes multiple servos responsible for facial movement, including areas such as:

- mouth,
- jaw,
- eyes,
- eyebrows,
- facial expression elements.

The system maps emotional states to predefined servo positions. For example, happy, sad, surprise, and neutral expressions are represented by different servo angle configurations.

Servo limits are defined to protect the mechanical structure and prevent unsafe movement.

## Module Descriptions

## Vision Module

The `vision-module` contains the computer vision and Raspberry Pi runtime code.

Main responsibilities:

- camera input,
- face detection,
- face recognition,
- emotion recognition,
- Raspberry Pi camera diagnostics,
- communication with the speech module,
- servo control integration,
- facial movement logic.

Important areas:

```text
vision-module/src/rpi/
vision-module/src/detection/
vision-module/model_training/
vision-module/preprocessing/
vision-module/documentation/
vision-module/images/
```

The main Raspberry Pi runtime is located in:

```text
vision-module/src/rpi/main_vision_network.py
```

This file combines the vision pipeline with robot control logic. It handles face visibility, identity, emotion, expression mapping, and communication with the rest of the system.

## Speech Module

The `speech-module` contains the Jetson-side speech and NLP pipeline.

Main responsibilities:

- speech-to-text,
- intent detection,
- entity extraction,
- local LLM interaction,
- text-to-speech,
- conversation memory,
- user identity handling,
- weather responses,
- room and teacher lookup,
- auditorium and special room lookup,
- communication with the Raspberry Pi.

Important files and directories:

```text
speech-module/nlp_pipeline.py
speech-module/nlp_pipeline_jetson_test.py
speech-module/utils/
speech-module/PiperTTS/
speech-module/db_specialrooms.json
speech-module/imiona_polskie.txt
```

The Jetson test pipeline is located in:

```text
speech-module/nlp_pipeline_jetson_test.py
```

This file orchestrates the main speech interaction loop. It receives data from the vision system, listens to the user, generates a response, speaks the answer, and sends mouth movement data back to the Raspberry Pi.

## Hardware Module

The `hardware-module` contains the mechanical and electronics-related project files.

Main responsibilities:

- 3D model storage,
- printed head structure documentation,
- servo placement documentation,
- circuit documentation,
- servo testing scripts,
- jaw synchronization experiments,
- face animation prototypes,
- hardware reports.

Important areas:

```text
hardware-module/3Dmodel/
hardware-module/Code/
hardware-module/DescServos/
hardware-module/Raports/
hardware-module/Circuit Diagram.pdf
```

The hardware module documents how the physical robot head is built and how the servos are assigned to facial movements.

## Data Flow

The project uses the following high-level data flow:

```text
Camera frame
   ↓
Face detection
   ↓
Face recognition and emotion recognition
   ↓
Identity and emotion sent to speech module
   ↓
User speaks
   ↓
Speech-to-text
   ↓
Intent detection and entity extraction
   ↓
Database lookup or local LLM response
   ↓
Text-to-speech
   ↓
Audio playback
   ↓
Mouth movement signal sent to Raspberry Pi
   ↓
Servo-controlled facial movement
```

## Communication Between Devices

The Raspberry Pi and Jetson communicate through HTTP endpoints over the local network.

The Raspberry Pi sends visual context to the Jetson, including:

- face visibility,
- recognized identity,
- detected emotion.

The Jetson sends mouth movement information to the Raspberry Pi, including:

- current speech volume level,
- jaw movement intensity,
- stop/reset mouth movement signal.

This split allows the Raspberry Pi to focus on real-time physical control while the Jetson handles heavier AI workloads.

## Conversation Logic

The conversation system is designed around real spoken interaction.

The assistant is prompted to behave as a physical robotic head rather than a text chatbot. It avoids responses that refer to typing, screens, chats, or written interaction.

The response generation logic considers:

- the current user,
- the detected emotion,
- the recent conversation history,
- the user’s latest question,
- whether the request should be answered from structured data or by the LLM.

The system also maintains a short conversation buffer so that responses can remain coherent during a session.

## Knowledge and Utility Features

The speech module includes several utility systems for specific types of questions.

### Room Search

The robot can answer questions about rooms by extracting room identifiers and searching local room data.

### Teacher Search

The robot can search for teacher information and provide room-related responses when the teacher exists in the local database.

### Auditorium and Special Room Search

The robot can respond to questions about auditoriums and selected special locations such as libraries, cloakrooms, or cafeterias.

### Weather

The robot can provide weather-related responses through the weather utility module.

## Facial Expressions

The robot can physically represent selected emotional states using servo positions.

Supported or planned expressions include:

- neutral,
- happy,
- sad,
- surprise,
- angry,
- fear,
- disgust.

Each expression is represented by a set of servo angle targets. These targets move different parts of the robot face to create a visible expression.

The emotion system is connected to both:

- detected user emotion,
- generated conversational state.

## Hardware Design

The physical head is based on a 3D-printed structure with multiple servos installed inside the face.

The hardware documentation includes:

- printable mechanical components,
- base structure,
- servo mapping,
- circuit diagram,
- assembly-related photos,
- test scripts,
- reports.

The servo system is designed to control facial motion while respecting mechanical limits.

## Technology Stack

### Hardware

- Raspberry Pi 5 8 GB
- NVIDIA Jetson Orin Nano Super 8 GB
- Raspberry Pi camera
- microphone
- speaker
- servo controller
- MG90s servos
- MG996r servos
- NVMe SSD
- DC-DC converters
- 3D-printed parts
- mechanical fasteners

### Software

- Python
- TensorFlow
- PyTorch
- OpenCV
- MediaPipe
- BlazingFace
- Whisper
- Faster Whisper
- Piper TTS
- GLiNER
- LangChain
- Ollama
- Flask
- PyAudio
- speech_recognition
- NumPy
- scikit-learn

## Development Branches

The project was developed across multiple branches.

### `master`

The main integration branch. It contains the merged version of the project with all major modules combined.

### `vision-module`

Contains the vision subsystem, including camera processing, recognition models, Raspberry Pi scripts, and visual-to-physical behavior integration.

### `speech-module`

Contains the speech and NLP subsystem, including STT, intent detection, GLiNER extraction, LLM response generation, TTS, and Jetson-side runtime logic.

### `hardware-module`

Contains the mechanical, electrical, and servo-related work, including 3D models, circuit diagrams, servo documentation, and hardware test code.

## Project Milestones

The project was planned around several technical milestones:

1. Building and validating the physical robot head structure.
2. Preparing power, servo, and mechanical control systems.
3. Creating a vision pipeline for recognition and emotion detection.
4. Creating a speech pipeline with STT, LLM, and TTS.
5. Running AI workloads across Raspberry Pi and Jetson hardware.
6. Integrating vision, speech, and facial actuation.
7. Testing real-time interaction with users.
8. Synchronizing jaw movement with generated speech.
9. Validating complete robot behavior in a controlled environment.

## Repository Language Composition

The repository is primarily written in Python, with additional Shell scripts and Jupyter Notebook content used for supporting tasks such as setup, testing, experimentation, or model-related work.

## Intended Use

This project is intended for research, prototyping, and educational development of an interactive humanoid robotic head.

It demonstrates how multiple AI and robotics components can be integrated into one physical system:

- perception,
- speech interaction,
- language understanding,
- local AI inference,
- mechanical expression,
- embedded device coordination.

## Notes

This project depends on correct hardware assembly, calibrated servos, configured camera and audio devices, local network connectivity between devices, and available machine learning model files.

Because the robot controls physical servos, all mechanical limits and power requirements should be checked carefully during development and testing.
