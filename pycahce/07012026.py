import cv2
import numpy as np
import streamlit as st
from PIL import Image
from collections import defaultdict
import io

# ============================================================================
# STANDARD BRAILLE ALPHABET MAPPING (Grade 1)
# ============================================================================

# Position notation:
#   1  4
#   2  5
#   3  6

BRAILLE_TO_LETTER = {
    (1,): 'A',
    (1, 2): 'B',
    (1, 4): 'C',
    (1, 4, 5): 'D',
    (1, 5): 'E',
    (1, 2, 4): 'F',
    (1, 2, 4, 5): 'G',
    (1, 2, 5): 'H',
    (2, 4): 'I',
    (2, 4, 5): 'J',
    (1, 3): 'K',
    (1, 2, 3): 'L',
    (1, 3, 4): 'M',
    (1, 3, 4, 5): 'N',
    (1, 3, 5): 'O',
    (1, 2, 3, 4): 'P',
    (1, 2, 3, 4, 5): 'Q',
    (1, 2, 3, 5): 'R',
    (2, 3, 4): 'S',
    (2, 3, 4, 5): 'T',
    (1, 3, 6): 'U',
    (1, 2, 3, 6): 'V',
    (2, 4, 5, 6): 'W',
    (1, 3, 4, 6): 'X',
    (1, 3, 4, 5, 6): 'Y',
    (1, 3, 5, 6): 'Z',
    (): ' ',  # Empty cell
}


# ============================================================================
# SIMPLIFIED PREPROCESSING MODULE
# ============================================================================

def preprocess_image_simple(image):
    """
    Simplified preprocessing for clean digital images.

    Pipeline:
    1. Convert to grayscale
    2. Apply Otsu's global thresholding
    3. Invert binary image (dots become white on black)
    4. Apply minimal morphological operations

    Args:
        image: Input BGR image

    Returns:
        Binary image with white dots on black background
    """
    # Step 1: Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Step 2: Apply Otsu's automatic thresholding
    # Otsu's method automatically calculates optimal threshold
    _, binary = cv2.threshold(
        gray,
        0,  # Threshold value (ignored with OTSU flag)
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Step 3: Invert so dots are white (255) on black (0)
    # Standard braille images have black dots on white background
    inverted = cv2.bitwise_not(binary)

    # Step 4: Minimal morphological cleanup
    # Remove small noise using opening operation
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned = cv2.morphologyEx(inverted, cv2.MORPH_OPEN, kernel, iterations=1)

    return cleaned


# ============================================================================
# DOT DETECTION MODULE
# ============================================================================

class BrailleDot:
    """Represents a detected braille dot with geometric properties."""

    def __init__(self, contour):
        """
        Initialize dot from contour.

        Args:
            contour: OpenCV contour
        """
        self.contour = contour

        # Calculate centroid using moments
        M = cv2.moments(contour)
        if M["m00"] != 0:
            self.cx = int(M["m10"] / M["m00"])
            self.cy = int(M["m01"] / M["m00"])
        else:
            # Fallback to bounding box center
            x, y, w, h = cv2.boundingRect(contour)
            self.cx = x + w // 2
            self.cy = y + h // 2

        # Calculate bounding box
        self.x, self.y, self.w, self.h = cv2.boundingRect(contour)

        # Calculate area
        self.area = cv2.contourArea(contour)

    def __repr__(self):
        return f"Dot(cx={self.cx}, cy={self.cy}, area={self.area:.0f})"


def detect_dots(binary_image):
    """
    Detect all braille dots using contour detection.

    Algorithm:
    1. Find contours in binary image
    2. Filter by area (remove noise and large objects)
    3. Filter by circularity (dots should be roughly circular)
    4. Create BrailleDot objects

    Args:
        binary_image: Binary image with white dots on black background

    Returns:
        List of BrailleDot objects
    """
    # Find all contours
    contours, _ = cv2.findContours(
        binary_image,
        cv2.RETR_EXTERNAL,  # Only external contours
        cv2.CHAIN_APPROX_SIMPLE
    )

    dots = []

    # Calculate dynamic area thresholds based on image size
    image_area = binary_image.shape[0] * binary_image.shape[1]
    min_area = max(10, int(image_area * 0.00001))  # 0.001% of image
    max_area = max(5000, int(image_area * 0.01))  # 1% of image

    for contour in contours:
        area = cv2.contourArea(contour)

        # Filter by area
        if area < min_area or area > max_area:
            continue

        # Filter by circularity (optional, helps with noisy images)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = (4 * np.pi * area) / (perimeter ** 2)

        # Accept dots with reasonable circularity
        # Relaxed threshold for imperfect circles
        if circularity > 0.3:
            dots.append(BrailleDot(contour))

    return dots


# ============================================================================
# SPATIAL SORTING MODULE
# ============================================================================

def sort_dots_spatially(dots, row_tolerance=20):
    """
    Sort dots top-to-bottom, left-to-right for reading order.

    Algorithm:
    1. Group dots into rows (within row_tolerance pixels vertically)
    2. Sort rows by average y-coordinate
    3. Within each row, sort left-to-right by x-coordinate

    Args:
        dots: List of BrailleDot objects
        row_tolerance: Vertical distance to consider dots in same row

    Returns:
        Sorted list of dots in reading order
    """
    if len(dots) == 0:
        return []

    # Sort by y-coordinate first
    dots_by_y = sorted(dots, key=lambda d: d.cy)

    # Group into rows
    rows = []
    current_row = [dots_by_y[0]]

    for dot in dots_by_y[1:]:
        if abs(dot.cy - current_row[-1].cy) <= row_tolerance:
            current_row.append(dot)
        else:
            rows.append(current_row)
            current_row = [dot]

    rows.append(current_row)

    # Sort each row left-to-right
    sorted_dots = []
    for row in rows:
        row_sorted = sorted(row, key=lambda d: d.cx)
        sorted_dots.extend(row_sorted)

    return sorted_dots


# ============================================================================
# GAP-BASED LETTER CLUSTERING MODULE
# ============================================================================

def cluster_dots_into_letters(sorted_dots, gap_multiplier=2.0):
    """
    Group dots into letter clusters using horizontal gap detection.

    Algorithm:
    1. Calculate horizontal distances between consecutive dots
    2. Find median gap (typical within-letter spacing)
    3. Significant gaps (> gap_multiplier × median) indicate letter boundaries
    4. Group dots between significant gaps into letter clusters

    Args:
        sorted_dots: Spatially sorted list of dots
        gap_multiplier: Multiplier for median gap to detect letter boundaries

    Returns:
        List of letter clusters (each cluster is a list of dots)
    """
    if len(sorted_dots) == 0:
        return []

    if len(sorted_dots) == 1:
        return [[sorted_dots[0]]]

    # Calculate horizontal gaps between consecutive dots
    gaps = []
    for i in range(1, len(sorted_dots)):
        gap = sorted_dots[i].cx - sorted_dots[i - 1].cx
        if gap > 0:  # Only consider forward gaps
            gaps.append(gap)

    if len(gaps) == 0:
        return [[sorted_dots[0]]]

    # Calculate median gap (typical within-letter spacing)
    median_gap = np.median(gaps)

    # Determine threshold for significant gaps
    gap_threshold = median_gap * gap_multiplier

    # Cluster dots based on gaps
    clusters = []
    current_cluster = [sorted_dots[0]]

    for i in range(1, len(sorted_dots)):
        gap = sorted_dots[i].cx - sorted_dots[i - 1].cx

        if gap > gap_threshold:
            # Significant gap detected - start new letter
            clusters.append(current_cluster)
            current_cluster = [sorted_dots[i]]
        else:
            # Same letter - add to current cluster
            current_cluster.append(sorted_dots[i])

    # Add final cluster
    clusters.append(current_cluster)

    return clusters


# ============================================================================
# FIXED 2×3 GRID MAPPING MODULE
# ============================================================================

class BrailleLetter:
    """Represents a decoded braille letter with its properties."""

    def __init__(self, dots, positions, character):
        """
        Initialize braille letter.

        Args:
            dots: List of BrailleDot objects in this letter
            positions: Tuple of active positions (1-6)
            character: Decoded character
        """
        self.dots = dots
        self.positions = positions
        self.character = character

        # Calculate bounding box for entire letter
        if len(dots) > 0:
            self.bbox_x = min(d.x for d in dots)
            self.bbox_y = min(d.y for d in dots)
            self.bbox_x2 = max(d.x + d.w for d in dots)
            self.bbox_y2 = max(d.y + d.h for d in dots)
            self.bbox_w = self.bbox_x2 - self.bbox_x
            self.bbox_h = self.bbox_y2 - self.bbox_y
        else:
            self.bbox_x = self.bbox_y = self.bbox_w = self.bbox_h = 0
            self.bbox_x2 = self.bbox_y2 = 0

    def __repr__(self):
        return f"Letter('{self.character}', positions={self.positions})"


def map_cluster_to_grid(cluster):
    """
    Map a cluster of dots to 2×3 braille grid positions.

    Grid Layout:
        1  4   (Left column: 1,2,3 | Right column: 4,5,6)
        2  5
        3  6

    Algorithm:
    1. Calculate cluster bounding box
    2. Divide horizontally at midpoint (left vs right column)
    3. Divide vertically into thirds (top, middle, bottom rows)
    4. Assign each dot to position 1-6 based on centroid location

    Args:
        cluster: List of BrailleDot objects representing one letter

    Returns:
        BrailleLetter object with decoded character
    """
    if len(cluster) == 0:
        return BrailleLetter([], (), ' ')

    # Calculate cluster bounding box using centroids
    xs = [d.cx for d in cluster]
    ys = [d.cy for d in cluster]

    min_x = min(xs)
    max_x = max(xs)
    min_y = min(ys)
    max_y = max(ys)

    width = max_x - min_x
    height = max_y - min_y

    # Define grid boundaries
    mid_x = min_x + width / 2  # Column separator

    # Row separators (divide into thirds)
    row_1_max = min_y + height / 3
    row_2_max = min_y + 2 * height / 3

    # Map each dot to grid position
    positions_set = set()

    for dot in cluster:
        # Determine column (0 = left, 1 = right)
        col = 0 if dot.cx < mid_x else 1

        # Determine row (0 = top, 1 = middle, 2 = bottom)
        if dot.cy < row_1_max:
            row = 0
        elif dot.cy < row_2_max:
            row = 1
        else:
            row = 2

        # Map to braille position (1-6)
        if col == 0:  # Left column
            position = row + 1  # 1, 2, 3
        else:  # Right column
            position = row + 4  # 4, 5, 6

        positions_set.add(position)

    # Convert to sorted tuple
    positions = tuple(sorted(positions_set))

    # Look up character in braille alphabet
    character = BRAILLE_TO_LETTER.get(positions, '?')

    return BrailleLetter(cluster, positions, character)


# ============================================================================
# VISUALIZATION MODULE
# ============================================================================

def draw_letter_boxes(image, letters):
    """
    Draw bounding boxes around entire letters (not individual dots).

    Visualization:
    - Green rectangles around each letter
    - Character label above box
    - Position pattern below box
    - Red dots at centroids (for debugging)

    Args:
        image: Original image
        letters: List of BrailleLetter objects

    Returns:
        Annotated image
    """
    annotated = image.copy()

    for letter in letters:
        if letter.bbox_w == 0 or letter.bbox_h == 0:
            continue

        # Expand bounding box with padding
        padding = 5
        x1 = max(0, letter.bbox_x - padding)
        y1 = max(0, letter.bbox_y - padding)
        x2 = min(image.shape[1], letter.bbox_x2 + padding)
        y2 = min(image.shape[0], letter.bbox_y2 + padding)

        # Draw green rectangle around entire letter
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw character label above box
        label = letter.character
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        font_thickness = 2

        # Get text size for background
        (text_w, text_h), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )

        # Draw background for text
        cv2.rectangle(
            annotated,
            (x1, y1 - text_h - 10),
            (x1 + text_w + 10, y1),
            (0, 255, 0),
            -1
        )

        # Draw text
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 5),
            font,
            font_scale,
            (0, 0, 0),
            font_thickness
        )

        # Draw position pattern below box
        pattern_text = str(letter.positions)
        cv2.putText(
            annotated,
            pattern_text,
            (x1, y2 + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            1
        )

        # Draw red dots at centroids (for debugging)
        for dot in letter.dots:
            cv2.circle(annotated, (dot.cx, dot.cy), 3, (0, 0, 255), -1)

    return annotated


# ============================================================================
# MAIN PROCESSING PIPELINE
# ============================================================================

def process_braille_image(image):
    """
    Complete simplified braille recognition pipeline.

    Pipeline:
    1. Preprocess: Otsu thresholding + inversion
    2. Detect dots: Contour detection
    3. Sort spatially: Top-to-bottom, left-to-right
    4. Cluster: Gap-based letter grouping
    5. Decode: Fixed 2×3 grid mapping per letter
    6. Visualize: Letter-level bounding boxes

    Args:
        image: Input BGR image

    Returns:
        dict with keys:
            - 'text': Decoded text string
            - 'letters': List of BrailleLetter objects
            - 'dots': List of all BrailleDot objects
            - 'annotated': Annotated image
            - 'binary': Binary preprocessed image
    """
    # Step 1: Preprocess
    binary = preprocess_image_simple(image)

    # Step 2: Detect dots
    dots = detect_dots(binary)

    if len(dots) == 0:
        return {
            'text': '',
            'letters': [],
            'dots': [],
            'annotated': image.copy(),
            'binary': binary
        }

    # Step 3: Sort spatially
    sorted_dots = sort_dots_spatially(dots)

    # Step 4: Cluster into letters
    clusters = cluster_dots_into_letters(sorted_dots)

    # Step 5: Decode each letter
    letters = [map_cluster_to_grid(cluster) for cluster in clusters]

    # Step 6: Generate text output
    text = ''.join(letter.character for letter in letters)

    # Step 7: Visualize
    annotated = draw_letter_boxes(image, letters)

    return {
        'text': text,
        'letters': letters,
        'dots': dots,
        'annotated': annotated,
        'binary': binary
    }


# ============================================================================
# STREAMLIT WEB APPLICATION
# ============================================================================

def main():
    """
    Simplified Streamlit application for braille recognition.
    """
    st.set_page_config(
        page_title="Simplified Braille Recognition",
        page_icon="🔬",
        layout="wide"
    )

    # Custom CSS
    st.markdown("""
        <style>
        .main-title {
            font-size: 2.5rem;
            font-weight: bold;
            color: #2563eb;
            text-align: center;
            margin-bottom: 1rem;
        }
        .subtitle {
            text-align: center;
            color: #64748b;
            margin-bottom: 2rem;
        }
        .success-box {
            background-color: #dcfce7;
            border-left: 4px solid #22c55e;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        .info-box {
            background-color: #dbeafe;
            border-left: 4px solid #3b82f6;
            padding: 1rem;
            border-radius: 0.5rem;
            margin: 1rem 0;
        }
        </style>
    """, unsafe_allow_html=True)

    # Header
    st.markdown('<p class="main-title">🔬 Simplified Braille Recognition System</p>',
                unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Optimized for Clean, High-Contrast Digital Images</p>',
                unsafe_allow_html=True)

    # Architecture explanation
    with st.expander("📐 System Architecture", expanded=False):
        st.markdown("""
        ### Re-Architected Pipeline

        **1. Simplified Preprocessing:**
        - ✅ **Otsu's Global Thresholding** (automatic threshold calculation)
        - ✅ **Binary Inversion** (dots become white on black)
        - ✅ **Minimal Morphology** (noise removal only)
        - ❌ Removed: Adaptive thresholding, multi-scale processing, complex filtering

        **2. Deterministic Dot Detection:**
        - ✅ Standard **contour detection**
        - ✅ Area and circularity filtering
        - ✅ Simple, fast, reliable

        **3. Gap-Based Letter Clustering:**
        - ✅ Sort dots **top-to-bottom, left-to-right**
        - ✅ Calculate **median horizontal spacing**
        - ✅ Detect significant gaps (**> 2× median**)
        - ✅ Group dots between gaps into letters
        - ❌ Removed: Complex relative clustering, dynamic cell parameters

        **4. Fixed 2×3 Grid Mapping:**
        - ✅ Divide letter bounding box into **2 columns × 3 rows**
        - ✅ Map dot centroids to positions **1-6**
        - ✅ Look up character in standard alphabet
        - ❌ Removed: Adaptive grid sizing, ratio-based positioning

        **5. Letter-Level Visualization:**
        - ✅ **Green boxes** around entire letters
        - ✅ Character labels and position patterns
        - ✅ Red centroid dots for debugging
        - ❌ Removed: Individual dot boxes, intensity coding

        ---

        **Grid Position Layout:**
        ```
          1  4   (Left column: 1,2,3 | Right column: 4,5,6)
          2  5
          3  6
        ```
        """)

    # Sidebar with settings
    with st.sidebar:
        st.header("⚙️ Processing Parameters")

        row_tolerance = st.slider(
            "Row Tolerance (pixels)",
            min_value=5,
            max_value=50,
            value=20,
            help="Vertical distance to consider dots in same row"
        )

        gap_multiplier = st.slider(
            "Gap Multiplier",
            min_value=1.5,
            max_value=3.0,
            value=2.0,
            step=0.1,
            help="Multiplier for median gap to detect letter boundaries"
        )

        st.divider()

        show_binary = st.checkbox("Show Binary Image", value=False)
        show_details = st.checkbox("Show Detection Details", value=False)

        st.divider()

        st.subheader("📖 Braille Reference")
        st.code("""
A (1):    B (12):   C (14):
●  ○      ●  ○      ●  ○
○  ○      ●  ○      ○  ●
○  ○      ○  ○      ○  ○

D (145):  E (15):   F (124):
●  ○      ●  ○      ●  ○
○  ●      ○  ●      ●  ○
○  ●      ○  ○      ○  ●
        """)

    # File uploader
    uploaded_file = st.file_uploader(
        "📤 Upload Braille Image",
        type=['png', 'jpg', 'jpeg', 'bmp'],
        help="Upload a clean, high-contrast braille image"
    )

    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if image is None:
            st.error("❌ Error loading image. Please upload a valid image file.")
            return

        # Display image info
        h, w = image.shape[:2]
        st.info(f"📐 Image Resolution: **{w} × {h}** pixels")

        # Process image
        with st.spinner("🔄 Processing image..."):
            # Use custom parameters
            result = process_braille_image(image)

            # Re-process with custom parameters if needed
            if gap_multiplier != 2.0 or row_tolerance != 20:
                binary = preprocess_image_simple(image)
                dots = detect_dots(binary)
                sorted_dots = sort_dots_spatially(dots, row_tolerance=row_tolerance)
                clusters = cluster_dots_into_letters(sorted_dots, gap_multiplier=gap_multiplier)
                letters = [map_cluster_to_grid(cluster) for cluster in clusters]
                text = ''.join(letter.character for letter in letters)
                annotated = draw_letter_boxes(image, letters)

                result = {
                    'text': text,
                    'letters': letters,
                    'dots': dots,
                    'annotated': annotated,
                    'binary': binary
                }

        st.success("✅ Processing Complete!")

        # Display results
        col1, col2 = st.columns([1.3, 1])

        with col1:
            st.subheader("📷 Detected Letters")
            st.image(
                cv2.cvtColor(result['annotated'], cv2.COLOR_BGR2RGB),
                caption="Letter-level detection with bounding boxes",
                use_column_width=True
            )

            # Show binary if requested
            if show_binary:
                st.subheader("🔲 Binary Image (After Otsu)")
                st.image(
                    result['binary'],
                    caption="White dots on black background",
                    use_column_width=True
                )

        with col2:
            st.subheader("📝 Decoded Text")

            # Text output in success box
            if result['text']:
                st.markdown(f"""
                <div class="success-box">
                    <h3 style="margin-top: 0;">Recognized Text:</h3>
                    <p style="font-size: 2rem; font-family: monospace; letter-spacing: 0.2em; margin: 0;">
                        {result['text']}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.warning("⚠️ No text detected. Try adjusting parameters.")

            # Statistics
            st.subheader("📊 Statistics")
            stats_col1, stats_col2 = st.columns(2)

            with stats_col1:
                st.metric("Letters Detected", len(result['letters']))
                st.metric("Total Dots", len(result['dots']))

            with stats_col2:
                if result['text']:
                    char_count = len(result['text'])
                    st.metric("Characters", char_count)
                    unknown_count = result['text'].count('?')
                    st.metric("Unknown (?)", unknown_count)

        # Detailed analysis
        if show_details and len(result['letters']) > 0:
            st.divider()
            st.subheader("🔍 Detailed Detection Analysis")

            # Create table data
            table_data = []
            for i, letter in enumerate(result['letters'], 1):
                table_data.append({
                    'Index': i,
                    'Character': letter.character,
                    'Positions': str(letter.positions),
                    'Dots': len(letter.dots),
                    'BBox': f"({letter.bbox_x}, {letter.bbox_y}, {letter.bbox_w}, {letter.bbox_h})"
                })

            st.table(table_data)

            # Show dot distribution
            st.subheader("📈 Dot Distribution")
            dots_per_letter = [len(letter.dots) for letter in result['letters']]
            st.bar_chart(dots_per_letter)

    else:
        # Instructions when no file uploaded
        st.markdown("""
        <div class="info-box">
            <h3>👆 Upload a braille image to begin</h3>
            <p><strong>This system is optimized for:</strong></p>
            <ul>
                <li>✅ Clean, high-contrast digital braille reference images</li>
                <li>✅ Standardized braille alphabet layouts</li>
                <li>✅ Images with clear letter spacing</li>
                <li>✅ Black dots on white background (or white on black)</li>
            </ul>
            <p><strong>Best results with:</strong></p>
            <ul>
                <li>📸 Digital braille charts/reference sheets</li>
                <li>💻 Computer-generated braille images</li>
                <li>🖨️ High-quality printed braille scans</li>
                <li>📱 Clear photos of braille displays</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Example workflow
        st.subheader("🔄 Processing Workflow")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**1️⃣ Preprocessing**")
            st.code("""
# Otsu thresholding
_, binary = cv2.threshold(
    gray, 0, 255,
    cv2.THRESH_BINARY + 
    cv2.THRESH_OTSU
)

# Invert
inverted = cv2.bitwise_not(binary)
            """, language='python')

        with col2:
            st.markdown("**2️⃣ Gap Detection**")
            st.code("""
# Calculate median gap
gaps = [dots[i].cx - dots[i-1].cx]
median_gap = np.median(gaps)

# Detect letter boundaries
if gap > median_gap * 2:
    start_new_letter()
            """, language='python')

        with col3:
            st.markdown("**3️⃣ Grid Mapping**")
            st.code("""
# Divide into 2×3 grid
mid_x = bbox_x + width / 2
row_1 = bbox_y + height / 3
row_2 = bbox_y + 2*height / 3

# Map to positions 1-6
position = get_grid_position(
    dot.cx, dot.cy
)
            """, language='python')


# ============================================================================
# APPLICATION ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()