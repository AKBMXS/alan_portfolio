OCR-to-Speech Android App (MyEyes)
An accessibility-focused Android app that converts real-time text captured through the camera into speech. Built to assist visually impaired users by providing instant audio feedback of printed text.

---

Project Overview
The app enables users to:
- Point the camera at any printed text  
- Extract text in real-time using OCR  
- Hear the text spoken aloud via Text-to-Speech  
- Navigate a simple interface designed for accessibility  

---

Architecture
1. Camera Input  
- Uses CameraX / Camera API for live preview  

2. OCR Engine  
- Google ML Kit (Text Recognition API)  
- Fast & lightweight inference on-device  

3. Speech Output 
- Android TextToSpeech Engine  
- Adjustable pitch/speed  

4. UI Layer  
- Large buttons  
- High-contrast UI  
- Simple workflow for ease of use  

---

Tools & Technologies
- Android Studio  
- Java  
- Google ML Kit OCR  
- Android TextToSpeech  
- XML UI layouts  

---

Key Contributions
- Built entire app using Java + ML Kit  
- Integrated OCR with real-time camera feed  
- Used TTS to convert extracted text into speech  
- Designed accessibility-first UI/UX  

---

Outcome
A fully functional accessibility tool enabling visually impaired users to read printed text instantly through speech output.

