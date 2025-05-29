import streamlit as st
import cv2
import face_recognition
import numpy as np
import os
from PIL import Image
import io
from datetime import datetime
import tempfile

class FaceSimilarityApp:
    def __init__(self):
        self.setup_page_config()
        
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="Face Similarity Detector",
            page_icon="👥",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
    def load_and_encode_face(self, image_file):
        """Load an image and encode the face"""
        try:
            # Handle different types of image inputs
            if hasattr(image_file, 'read'):
                # This is an uploaded file from Streamlit
                image_bytes = image_file.read()
                # Create a fresh BytesIO object
                image_io = io.BytesIO(image_bytes)
                image = Image.open(image_io)
                # Convert PIL image to RGB if it isn't already
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_np = np.array(image)
            elif isinstance(image_file, np.ndarray):
                # This is already a numpy array (e.g., from camera)
                image_np = image_file
                # Ensure it's RGB format
                if len(image_np.shape) == 3 and image_np.shape[2] == 3:
                    # Assume it's RGB already for numpy arrays
                    pass
            elif isinstance(image_file, Image.Image):
                # This is a PIL Image
                if image_file.mode != 'RGB':
                    image_file = image_file.convert('RGB')
                image_np = np.array(image_file)
            else:
                return None, f"Unsupported image type: {type(image_file)}"
            
            # Find face locations
            face_locations = face_recognition.face_locations(image_np)
            
            if len(face_locations) == 0:
                return None, "No face found in the image"
                
            if len(face_locations) > 1:
                st.warning(f"Multiple faces found. Using the first detected face.")
            
            # Get face encoding
            face_encodings = face_recognition.face_encodings(image_np, face_locations)
            
            if len(face_encodings) > 0:
                return face_encodings[0], f"Face detected successfully! ({len(face_locations)} face(s) found)"
            else:
                return None, "Could not encode the face"
                
        except Exception as e:
            return None, f"Error processing image: {str(e)}"
    
    def calculate_similarity(self, encoding1, encoding2):
        """Calculate similarity percentage between two face encodings"""
        if encoding1 is None or encoding2 is None:
            return 0
            
        # Calculate face distance (lower distance = more similar)
        face_distance = face_recognition.face_distance([encoding1], encoding2)[0]
        
        # Convert distance to similarity percentage
        similarity = max(0, (1 - face_distance) * 100)
        
        return similarity
    
    def get_similarity_status(self, similarity):
        """Get status message based on similarity percentage"""
        if similarity > 80:
            return "🟢 Very High Similarity - Likely the same person", "success"
        elif similarity > 60:
            return "🟡 High Similarity - Probably the same person", "warning"
        elif similarity > 40:
            return "🟠 Moderate Similarity - Could be the same person", "info"
        elif similarity > 20:
            return "🔴 Low Similarity - Probably different people", "error"
        else:
            return "⚫ Very Low Similarity - Likely different people", "error"
    
    def capture_camera_frame(self):
        """Capture image from camera with manual control"""
        try:
            camera = cv2.VideoCapture(0)
            
            if not camera.isOpened():
                return None, "Could not access camera"
            
            # Set camera properties for better quality
            camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            
            # Allow camera to warm up
            for _ in range(5):
                ret, frame = camera.read()
                if not ret:
                    camera.release()
                    return None, "Could not capture image"
            
            # Get the final frame
            ret, frame = camera.read()
            camera.release()
            
            if ret:
                # Convert BGR to RGB for display
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                return frame_rgb, "Image captured successfully"
            else:
                return None, "Could not capture image"
                
        except Exception as e:
            return None, f"Camera error: {str(e)}"
    
    def display_comparison_result(self, similarity, image1, image2, status_msg1, status_msg2):
        """Display comparison results in a nice format"""
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            st.subheader("Image 1")
            if image1 is not None:
                st.image(image1, use_container_width=True)
                st.success(status_msg1)
            else:
                st.error("No image provided")
        
        with col2:
            st.subheader("Similarity Analysis")
            
            # Create a large similarity display
            st.markdown(f"""
            <div style="text-align: center; padding: 20px; background-color: #f0f2f6; border-radius: 10px; margin: 20px 0;">
                <h1 style="color: #1f77b4; font-size: 3em; margin: 0;">{similarity:.1f}%</h1>
                <p style="font-size: 1.2em; margin: 10px 0;">Face Similarity</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Status message
            status_text, status_type = self.get_similarity_status(similarity)
            if status_type == "success":
                st.success(status_text)
            elif status_type == "warning":
                st.warning(status_text)
            elif status_type == "info":
                st.info(status_text)
            else:
                st.error(status_text)
        
        with col3:
            st.subheader("Image 2")
            if image2 is not None:
                st.image(image2, use_container_width=True)
                st.success(status_msg2)
            else:
                st.error("No image provided")
    
    def run(self):
        """Main application"""
        # Header
        st.title(" Face Similarity Detector")
        st.markdown("Compare faces and get similarity percentage using advanced AI algorithms")
        st.markdown("---")
        
        # Sidebar for mode selection
        st.sidebar.title("🔧 Options")
        mode = st.sidebar.selectbox(
            "Select Mode:",
            ["📸 Camera + Reference Image", "🖼️ Compare Two Images", "ℹ️ About"]
        )
        
        if mode == "📸 Camera + Reference Image":
            self.camera_mode()
        elif mode == "🖼️ Compare Two Images":
            self.image_comparison_mode()
        else:
            self.about_page()
    
    def camera_mode(self):
        """Camera capture and comparison mode"""
        st.header("📸 Camera Capture Mode")
        st.markdown("Upload a reference image and capture from camera to compare")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Step 1: Upload Reference Image")
            reference_file = st.file_uploader(
                "Choose reference image file",
                type=['jpg', 'jpeg', 'png'],
                key="reference_upload"
            )
            
            if reference_file is not None:
                # Create a copy of the image for display
                reference_file.seek(0)
                reference_image = Image.open(reference_file)
                st.image(reference_image, caption="Reference Image", use_container_width=True)
                
                # Encode reference face
                reference_file.seek(0)  # Reset file pointer before processing
                ref_encoding, ref_status = self.load_and_encode_face(reference_file)
                if ref_encoding is not None:
                    st.success(ref_status)
                    # Store the encoding in session state to avoid reprocessing
                    st.session_state.ref_encoding = ref_encoding
                    st.session_state.ref_status = ref_status
                    st.session_state.ref_image = reference_image
                else:
                    st.error(ref_status)
                    return
        
        with col2:
            st.subheader("Step 2: Capture from Camera")
            
            if st.button("📷 Capture Photo", type="primary"):
                with st.spinner("Accessing camera..."):
                    captured_frame, capture_status = self.capture_camera_frame()
                    
                if captured_frame is not None:
                    st.image(captured_frame, caption="Captured Image", use_container_width=True)
                    st.success(capture_status)
                    
                    # Store captured image in session state
                    st.session_state.captured_image = captured_frame
                else:
                    st.error(capture_status)
        
        # Compare if both images are available
        if ('ref_encoding' in st.session_state and 
            'captured_image' in st.session_state):
            st.markdown("---")
            st.header("🔍 Comparison Results")
            
            if st.button(" Compare Faces", type="primary"):
                with st.spinner("Analyzing faces..."):
                    # Encode captured face
                    captured_encoding, captured_status = self.load_and_encode_face(st.session_state.captured_image)
                    
                    if st.session_state.ref_encoding is not None and captured_encoding is not None:
                        similarity = self.calculate_similarity(st.session_state.ref_encoding, captured_encoding)
                        
                        self.display_comparison_result(
                            similarity,
                            st.session_state.ref_image,
                            Image.fromarray(st.session_state.captured_image),
                            st.session_state.ref_status,
                            captured_status
                        )
                    else:
                        st.error("Could not process one or both images for comparison")
    
    def image_comparison_mode(self):
        """Compare two uploaded images mode"""
        st.header("🖼️ Image Comparison Mode")
        st.markdown("Upload two images to compare their facial similarity")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Image 1")
            image1_file = st.file_uploader(
                "Choose first image",
                type=['jpg', 'jpeg', 'png'],
                key="image1_upload"
            )
            
            if image1_file is not None:
                image1_file.seek(0)
                image1 = Image.open(image1_file)
                st.image(image1, caption="Image 1", use_container_width=True)
        
        with col2:
            st.subheader("Image 2")
            image2_file = st.file_uploader(
                "Choose second image",
                type=['jpg', 'jpeg', 'png'],
                key="image2_upload"
            )
            
            if image2_file is not None:
                image2_file.seek(0)
                image2 = Image.open(image2_file)
                st.image(image2, caption="Image 2", use_container_width=True)
        
        # Compare button
        if image1_file is not None and image2_file is not None:
            st.markdown("---")
            
            if st.button(" Compare Faces", type="primary", key="compare_images"):
                with st.spinner("Analyzing faces..."):
                    # Reset file pointers before processing
                    image1_file.seek(0)
                    image2_file.seek(0)
                    
                    # Encode both faces
                    encoding1, status1 = self.load_and_encode_face(image1_file)
                    
                    # Reset file pointer again for second processing
                    image2_file.seek(0)
                    encoding2, status2 = self.load_and_encode_face(image2_file)
                    
                    if encoding1 is not None and encoding2 is not None:
                        similarity = self.calculate_similarity(encoding1, encoding2)
                        
                        st.header("🔍 Comparison Results")
                        # Use the images already loaded for display
                        image1_file.seek(0)
                        image2_file.seek(0)
                        self.display_comparison_result(
                            similarity,
                            Image.open(image1_file),
                            Image.open(image2_file),
                            status1,
                            status2
                        )
                    else:
                        st.error("Could not process one or both images:")
                        if encoding1 is None:
                            st.error(f"Image 1: {status1}")
                        if encoding2 is None:
                            st.error(f"Image 2: {status2}")
    
    def about_page(self):
        """About page with information"""
        st.header("ℹ️ About Face Similarity Detector")
        
        st.markdown("""
        ###  What it does
        This application uses advanced machine learning algorithms to compare facial features and determine similarity between faces.
        
        ###  How it works
        1. **Face Detection**: Locates faces in images using HOG (Histogram of Oriented Gradients)
        2. **Face Encoding**: Creates 128-dimensional face signatures using deep neural networks
        3. **Similarity Calculation**: Computes Euclidean distance between encodings
        4. **Percentage Conversion**: Converts distance to intuitive similarity percentage
        
        ###  Similarity Levels
        - **80%+**: Very High Similarity - Likely the same person
        - **60-80%**: High Similarity - Probably the same person
        - **40-60%**: Moderate Similarity - Could be the same person
        - **20-40%**: Low Similarity - Probably different people
        - **0-20%**: Very Low Similarity - Likely different people
        
        ###  Features
        - **Real-time camera capture**
        - **Multiple image format support** (JPG, PNG, JPEG)
        - **Advanced face recognition** using state-of-the-art algorithms
        - **User-friendly interface** with visual feedback
        - **Multiple comparison modes**
        
        ###  Privacy
        - All processing happens locally on your machine
        - No images are stored or sent to external servers
        - Camera access is only used when explicitly requested
        
        ###  Tips for best results
        - Use clear, well-lit images
        - Ensure faces are clearly visible and not obscured
        - Front-facing photos work best
        - Avoid extreme angles or lighting conditions
        """)
        
        st.markdown("---")
        st.markdown("**Built with:** Python, Streamlit, OpenCV, face_recognition, dlib")

def main():
    app = FaceSimilarityApp()
    app.run()

if __name__ == "__main__":
    main()