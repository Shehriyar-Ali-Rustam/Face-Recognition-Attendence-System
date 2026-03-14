# LinkedIn Post — AttendEase

---

🚀 **Just built AttendEase — an AI-powered Face Recognition Attendance System!**

Tired of manual roll calls, sign-in sheets, and proxy attendance? I built a full-stack solution that eliminates all of that using real-time face recognition.

---

**What it does:**

📸 **Instant Face Scan** — Students mark attendance in seconds by looking at the camera. No login, no ID card, no friction.

🧠 **AI Face Recognition** — Powered by `dlib` + `face_recognition` with HOG-based detection and ResNet encoding. Matches faces against trained models with < 0.5 distance threshold for accuracy.

🎓 **Student Portal** — Dashboard with attendance stats, history timeline, profile management, and face registration status.

🛡️ **Admin Portal** — Register students, capture face images, train the recognition model, manage records, and generate reports — all from one place.

🔐 **Secure Auth** — SHA-256 password hashing, email-based password reset with OTP verification, and Google OAuth support.

---

**Tech Stack:**

- **Python** + **Streamlit** — full web app, zero frontend framework needed
- **face_recognition** + **dlib** — real-time face detection and encoding
- **OpenCV** — live webcam feed with bounding box overlays
- **SQLAlchemy** + **SQLite** — relational database for students, users, and attendance records
- **Custom CSS** — dark glassmorphism UI with backdrop-filter blur, split-screen login, and animated glass cards

---

**Key Features:**

✅ Real-time webcam face scanning with live bounding box feedback
✅ Model training pipeline — capture → train → deploy in one click
✅ Attendance marked with confidence score logged to database
✅ Duplicate attendance prevention (one mark per day per student)
✅ Email OTP password reset flow
✅ Quick Attendance mode — no login required for students
✅ Full admin CRUD — add, edit, delete students and their face data
✅ Professional dark UI with hero background and glassmorphism cards

---

**What I learned:**

Building this taught me a lot about the gap between ML models and production-ready apps. Getting `face_recognition` to work reliably on Python 3.12 (the `pkg_resources` deprecation broke the models package — had to patch it manually), managing SQLAlchemy schema migrations on a live SQLite database, and making Streamlit look like a real product with pure CSS injection were the biggest challenges.

The camera pipeline runs a 30-second recognition loop using `cv2.VideoCapture(0)` — each frame is flipped, converted to RGB, and passed through HOG face detection before encoding comparison. It's satisfying to see it light up green on a match.

---

**GitHub:** [github.com/Shehriyar-Ali-Rustam/Face-Recognition-Attendence-System](https://github.com/Shehriyar-Ali-Rustam/Face-Recognition-Attendence-System)

---

*Open to feedback, collaboration, and opportunities in AI/ML and full-stack development.*

**#Python #MachineLearning #ComputerVision #FaceRecognition #AI #OpenCV #Streamlit #FullStack #StudentProject #BuildInPublic #OpenSource**
