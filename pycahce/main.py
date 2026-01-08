import cv2
import streamlit as st
import pandas as pd
from datetime import datetime
import uuid
import os
from PIL import Image
from pathlib import Path

from BrailleImage import BrailleImage
from SegmentationEngine import SegmentationEngine
from BrailleClassifier import BrailleClassifier


def save_feedback_to_excel(feedback_entry, filename="braille_feedback.xlsx"):
    """Append feedback to Excel file - for admin/researcher use only"""
    try:
        # Check if file exists
        if os.path.exists(filename):
            # Load existing data
            df_existing = pd.read_excel(filename)
            # Append new entry
            df_new = pd.DataFrame([feedback_entry])
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            # Create new DataFrame with headers
            df_combined = pd.DataFrame([feedback_entry])

        # Save to Excel
        df_combined.to_excel(filename, index=False, engine='openpyxl')
        return True
    except Exception as e:
        print(f"Error saving to Excel: {str(e)}")
        return False


def main():
    # === Step 1: Get image path from user or command-line ===
    ''' if len(sys.argv) > 1:
        image_path = sys.argv[1]
    else:
        image_path = input("Enter the path to your Braille image: ").strip()

    print(f"[INFO] Loading image: {image_path}")
    try:
        braille_img = BrailleImage(image_path)
    except IOError as e:
        print(f"[ERROR] {e}")
        return

    # === Step 2: Initialize the segmentation engine ===
    print("[INFO] Detecting Braille dots and segmenting characters...")
    segmenter = SegmentationEngine(braille_img)

    # === Step 3: Initialize classifier ===
    classifier = BrailleClassifier()

    # === Step 4: Process Braille characters ===
    char_count = 0
    for character in segmenter:
        if character.is_valid():
            char_count += 1
            character.mark()           # Draw bounding box (optional)
            classifier.push(character) # Classify Braille pattern
        else:
            print(f"[WARN] Invalid character #{char_count+1} skipped.")

    # === Step 5: Display results ===
    result_text = classifier.digest()
    print("\n✅ Translated Braille text:\n")
    print(result_text)
    print(f"\n[INFO] Total characters detected: {char_count}")
    # === Step 6: Ask if user wants to save output ===
    save_choice = input("\nDo you want to save the translated text to a file? (y/n): ").strip().lower()
    if save_choice == "y":
        output_file = input("Enter output file name (e.g., output.txt): ").strip()
        if not output_file.endswith(".txt"):
            output_file += ".txt"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(result_text)
        print(f"[INFO] Translated text saved to {output_file}")

    # === Step 7: Show final annotated image ===
    annotated = braille_img.get_final_image()
    cv2.imshow("Braille Segmentation Result", annotated)
    print("[INFO] Press any key in the image window to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''# Page configuration
    st.set_page_config(
        page_title="Braille Image Translator",
        page_icon="iium_logo.png",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Custom CSS for beautiful styling
    st.markdown("""
        <style>
        .main {
            padding: 2rem;
        }
        .stApp {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .upload-section {
            background: white;
            padding: 2rem;
            border-radius: 15px;
            box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        }
        .result-box {
            background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
            border-left: 4px solid #667eea;
            padding: 1.5rem;
            border-radius: 10px;
            font-family: 'Monaco', monospace;
            font-size: 1.2rem;
            margin: 1rem 0;
        }
        .image-container {
            background: white;
            padding: 1rem;
            border-radius: 10px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
            margin: 1rem 0;
        }
        h1 {
            color: white;
            text-align: center;
            font-size: 3rem;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        }
        h2, h3 {
            color: white;
        }
        .stButton>button {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            padding: 0.75rem 2rem;
            border-radius: 10px;
            font-size: 1rem;
            font-weight: 600;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
            width: 100%;
        }
        .stButton>button:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
        }
        .info-box {
            background: rgba(255, 255, 255, 0.95);
            padding: 1.5rem;
            border-radius: 10px;
            margin: 1rem 0;
            border-left: 4px solid #38b2ac;
        }
        </style>
    """, unsafe_allow_html=True)

    # Helper function to convert OpenCV image to PIL
    def cv2_to_pil(cv2_image):
        """Convert OpenCV BGR image to PIL RGB image"""
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_image)

    # Helper function to save uploaded file temporarily
    def save_uploaded_file(uploaded_file):
        """Save uploaded file to temporary location and return path"""
        temp_path = Path("temp_uploads")
        temp_path.mkdir(exist_ok=True)

        file_path = temp_path / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        return str(file_path)

    # Main processing function
    def process_braille_image(image_path):
        """Process Braille image and return results"""
        try:
            # Initialize your existing classes
            braille_img = BrailleImage(image_path)
            segmentation_engine = SegmentationEngine(braille_img)
            classifier = BrailleClassifier()

            # Process characters
            character_count = 0
            for character in segmentation_engine:
                if character.is_valid():
                    character.mark()  # Mark on image
                    classifier.push(character)
                    character_count += 1

            # Get results
            translated_text = classifier.digest()

            # Get processed images
            original_img = braille_img.get_original_image()
            binary_img = braille_img.get_binary_image()
            edged_img = braille_img.get_edged_binary_image()
            final_img = braille_img.get_final_image()

            return {
                'success': True,
                'text': translated_text,
                'character_count': character_count,
                'original': original_img,
                'binary': binary_img,
                'edged': edged_img,
                'annotated': final_img
            }

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    # Title and header
    col_left, col_logo, col_title, col_right = st.columns([1, 1, 8, 1])

    with col_left:
        st.write("")

    with col_logo:
        try:
            # Add vertical spacing to align with title
            st.markdown("<div style='padding-top: 50px;'>", unsafe_allow_html=True)
            st.image("iium_logo.png", width=80)
            st.markdown("</div>", unsafe_allow_html=True)
        except:
            st.write("")

    with col_title:
        st.markdown("""
            <h1 style='margin-top: 30px; color: white; text-shadow: 2px 2px 4px rgba(0,0,0,0.3); white-space: nowrap;'>
                 Braille Image Translator
            </h1>
            <p style='color: white; font-size: 1.2rem; margin: 0;'>
                Transform Braille images into readable text instantly
            </p>
            <p style='color: white; font-size: 1rem; margin-top: 5px;'>
                International Islamic University Malaysia (IIUM)
            </p>
        """, unsafe_allow_html=True)

    with col_left:
        st.write("")

    with col_right:
        st.write("")

    st.markdown(""" <p style='color: white; font-size: 1rem; margin-top: 0px;'>
               Final Year Project : Development of Malay Braille Recognition System
            </p>
        """, unsafe_allow_html=True)
    # Info box
    st.markdown("""
        <div class="info-box">
            <strong>💡 How it works:</strong> Upload a clear image of Braille text. 
            Our system will detect the dots, segment characters, and translate them into readable text.
        </div>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        st.info("Upload a Braille image to begin translation")

        st.markdown("---")

        st.subheader("📊 Statistics")
        if 'stats' in st.session_state:
            st.metric("Characters Detected", st.session_state.stats.get('character_count', 0))
            st.metric("Images Processed", st.session_state.stats.get('images_processed', 0))

        st.markdown("---")

        st.subheader("ℹ️ About")
        st.write("""
        This Braille translator uses computer vision to:
        - Detect Braille dots
        - Segment characters
        - Translate to text
        """)

        st.markdown("---")

        if st.button("🔄 Reset App"):
            st.session_state.clear()
            st.rerun()

    # Initialize session state
    if 'stats' not in st.session_state:
        st.session_state.stats = {
            'images_processed': 0,
            'character_count': 0
        }

    # Main content area
    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        # File uploader
        uploaded_file = st.file_uploader(
            "📤 Upload Braille Image",
            type=['png', 'jpg', 'jpeg'],
            help="Upload a clear image of Braille text"
        )

        if uploaded_file is not None:
            # Display uploaded image
            st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

            # Process button
            if st.button("🚀 Translate Braille", use_container_width=True):
                with st.spinner("🔍 Processing your Braille image..."):
                    # Save uploaded file
                    image_path = save_uploaded_file(uploaded_file)

                    # Process image
                    result = process_braille_image(image_path)

                    if result['success']:
                        st.success("✅ Translation completed successfully!")

                        # Update stats
                        st.session_state.stats['images_processed'] += 1
                        st.session_state.stats['character_count'] = result['character_count']

                        # Display translated text
                        st.markdown("### ✨ Translated Text")
                        st.markdown(f"""
                            <div class="result-box">
                                {result['text'] if result['text'] else "No text detected"}
                            </div>
                        """, unsafe_allow_html=True)

                        # Display images in tabs
                        st.markdown("### 🖼️ Image Analysis")

                        tab1, tab2, tab3, tab4 = st.tabs([
                            "📷 Original",
                            "⚫ Binary",
                            "🔲 Edge Detection",
                            "🎯 Annotated"
                        ])

                        with tab1:
                            st.image(cv2_to_pil(result['original']),
                                     caption="Original Image",
                                     use_container_width=True)

                        with tab2:
                            st.image(result['binary'],
                                     caption="Binary Image (Content Detection)",
                                     use_container_width=True,
                                     channels="GRAY")

                        with tab3:
                            st.image(result['edged'],
                                     caption="Edge Detection Binary Image",
                                     use_container_width=True,
                                     channels="GRAY")

                        with tab4:
                            st.image(cv2_to_pil(result['annotated']),
                                     caption="Detected Characters (with bounding boxes)",
                                     use_container_width=True)

                        # Character count
                        st.info(f"🔢 Detected **{result['character_count']}** Braille characters")

                        # Download button for translated text
                        st.download_button(
                            label="📥 Download Translated Text",
                            data=result['text'],
                            file_name="translated_braille.txt",
                            mime="text/plain"
                        )

                    else:
                        st.error(f"❌ Error processing image: {result['error']}")

        else:
            # Placeholder when no file uploaded
            st.markdown("""
                <div class="upload-section" style="text-align: center; padding: 3rem;">
                    <h2 style="color: #667eea;">📤 Drop your Braille image here</h2>
                    <p style="color: #718096;">or use the upload button above</p>
                    <p style="color: #a0aec0; font-size: 0.9rem;">Supported formats: PNG, JPG, JPEG</p>
                </div>
            """, unsafe_allow_html=True)
    # Feedback Section
    # Feedback Section (Data saved automatically for research)
    st.markdown("---")
    st.markdown("### 📝 Maklum Balas & Penilaian Ketepatan / Feedback & Accuracy Rating")
    st.markdown(
        "<p style='color: rgba(255,255,255,0.8);'>Bantu kami memperbaiki penterjemah Braille dengan menilai ketepatan dan melaporkan sebarang terjemahan yang tidak betul.</p>",
        unsafe_allow_html=True)
    st.markdown(
        "<p style='color: rgba(255,255,255,0.8);'>Help us improve the Braille translator by rating the accuracy and reporting any incorrect translations.</p>",
        unsafe_allow_html=True)

    # Create two columns for feedback
    feedback_col1, feedback_col2 = st.columns([1, 1])

    with feedback_col1:
        st.markdown("#### ⭐ Nilai Ketepatan Terjemahan / Rate Translation Accuracy")

        # Star rating
        rating = st.radio(
            "Sejauh manakah ketepatan terjemahan? / How accurate was the translation?",
            options=[5, 4, 3, 2, 1],
            format_func=lambda x: "⭐" * x + "☆" * (5 - x) + f" ({x}/5)",
            index=0,
            key="accuracy_rating"
        )

        # Optional comment
        rating_comment = st.text_area(
            "Komen tambahan (pilihan) / Additional comments (optional)",
            placeholder="Beritahu kami lebih lanjut... / Tell us more about your experience...",
            max_chars=500,
            key="rating_comment"
        )

        if st.button("📤 Hantar Penilaian / Submit Rating", type="secondary", use_container_width=True):
            feedback_entry = {
                "Feedback ID": str(uuid.uuid4())[:8],
                "Type": "Rating",
                "Rating": rating,
                "Comment": rating_comment if rating_comment else "",
                "Incorrect Word": "",
                "Correct Word": "",
                "Context": "",
                "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }

            # Save to Excel file (only admin will access this)
            if save_feedback_to_excel(feedback_entry):
                st.success(f"✅ Terima kasih! Penilaian dihantar / Thank you! Rating submitted: {'⭐' * rating}")
                st.balloons()
            else:
                st.warning(
                    "⚠️ Maklum balas direkodkan tetapi tidak dapat disimpan ke fail / Feedback recorded but could not save to file")

    with feedback_col2:
        st.markdown("#### 🔤 Laporkan Terjemahan Yang Salah / Report Incorrect Translation")
        st.markdown(
            "<p style='color: rgba(255,255,255,0.7); font-size: 0.9rem;'>Jumpa perkataan yang tidak diterjemah dengan betul? Beritahu kami! / Found a word that was not translated correctly? Let us know!</p>",
            unsafe_allow_html=True)

        # Incorrect word input
        incorrect_word = st.text_input(
            "Perkataan yang salah / Incorrect word",
            placeholder="contoh / e.g., 'hlelo' (salah/wrong)",
            key="incorrect_word"
        )

        # Correct word input
        correct_word = st.text_input(
            "Patut jadi apa? / What should it be?",
            placeholder="contoh / e.g., 'hello' (betul/correct)",
            key="correct_word"
        )

        # Additional context
        word_context = st.text_area(
            "Konteks atau nota (pilihan) / Context or notes (optional)",
            placeholder="Berikan konteks tambahan... / Provide any additional context...",
            max_chars=300,
            key="word_context"
        )

        if st.button("📝 Hantar Pembetulan / Submit Correction", type="primary", use_container_width=True):
            if incorrect_word and correct_word:
                feedback_entry = {
                    "Feedback ID": str(uuid.uuid4())[:8],
                    "Type": "Correction",
                    "Rating": 0,
                    "Comment": "",
                    "Incorrect Word": incorrect_word,
                    "Correct Word": correct_word,
                    "Context": word_context if word_context else "",
                    "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }

                # Save to Excel file (only admin will access this)
                if save_feedback_to_excel(feedback_entry):
                    st.success(f"✅ Pembetulan dihantar / Correction submitted: '{incorrect_word}' → '{correct_word}'")
                    st.balloons()
                else:
                    st.warning(
                        "⚠️ Maklum balas direkodkan tetapi tidak dapat disimpan ke fail / Feedback recorded but could not save to file")
            else:
                st.warning("⚠️ Sila isi kedua-dua perkataan / Please fill in both the incorrect and correct words.")

    # Privacy notice
    st.markdown("---")
    st.markdown(
        "<p style='text-align: center; color: rgba(255,255,255,0.5); font-size: 0.8rem;'>🔒 Maklum balas anda digunakan untuk penambahbaikan sistem sahaja / Your feedback is used for system improvement only</p>",
        unsafe_allow_html=True)
    # Footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; color: white; padding: 1rem;">
            <p>Built with ❤️ using Streamlit | Powered by OpenCV & Computer Vision</p>
        </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()