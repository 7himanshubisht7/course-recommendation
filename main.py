import streamlit as st
import pickle
import pandas as pd
import os
from difflib import get_close_matches

# File paths (update these only if your files are elsewhere)
COURSE_DATA_PATH = "Coursera.csv"
SIMILARITY_PATH = "similarity.pkl"

# Load CSV and Pickle files safely
@st.cache_data
def load_data(course_data_path, similarity_path):
    try:
        if not os.path.exists(course_data_path):
            raise FileNotFoundError(f"{course_data_path} not found. Make sure the file is in the project directory.")
        if not os.path.exists(similarity_path):
            raise FileNotFoundError(f"{similarity_path} not found. Make sure the file is in the project directory.")

        course_data = pd.read_csv(course_data_path)
        with open(similarity_path, 'rb') as file:
            similarity_matrix = pickle.load(file)

        return course_data, similarity_matrix
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None

# Fuzzy match course name
def get_closest_course_name(user_input, course_data):
    course_data['course_name_lower'] = course_data['Course Name'].str.lower()
    matches = get_close_matches(user_input.lower(), course_data['course_name_lower'].tolist(), n=1, cutoff=0.6)
    return matches[0] if matches else None

# Recommend similar courses
def recommend_courses(course, course_data, similarity_matrix):
    course_data['course_name_lower'] = course_data['Course Name'].str.lower()
    if course not in course_data['course_name_lower'].values:
        return None

    index = course_data[course_data['course_name_lower'] == course].index[0]
    distances = similarity_matrix[index]
    similar_courses = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    recommendations = []
    for i in similar_courses:
        recommended_course = course_data.iloc[i[0]]
        recommendations.append({
            'course_name': recommended_course['Course Name'],
            'course_url': recommended_course['Course URL'],
            'course_description': recommended_course['Course Description'],
            'university': recommended_course['University']
        })
    return recommendations

# Display recommendations in card style
def display_course_cards(courses):
    for course in courses:
        st.markdown(f"""
        <div style="border:1px solid #e0e0e0; border-radius:10px; padding:15px; margin:10px 0; background-color:#f8f9fa;">
            <h4 style="color:#2c3e50;">
                <a href="{course['course_url']}" target="_blank" style="text-decoration:none;">
                    {course['course_name']}
                </a>
            </h4>
            <p style="color:#34495e;"><strong>University:</strong> {course['university']}</p>
            <p style="color:#555;">{course['course_description']}</p>
        </div>
        """, unsafe_allow_html=True)

# Main App
def main():
    st.set_page_config(page_title="Smart Course Recommender", layout="wide")
    st.markdown("<h1 style='text-align:center; color:#0072B5;'>🎓 Smart Course Recommendation System</h1>", unsafe_allow_html=True)

    # Load course and similarity data
    course_data, similarity_matrix = load_data(COURSE_DATA_PATH, SIMILARITY_PATH)
    if course_data is None or similarity_matrix is None:
        st.error("Could not load required data. Please ensure 'Coursera.csv' and 'similarity.pkl' are in the app directory.")
        return

    # Tabs
    tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔍 Recommend", "ℹ️ About"])

    # ----------- HOME TAB -----------
    with tab1:
        st.markdown("""
        <div style="padding: 20px; background-color: #eaf4fc; border-radius: 10px;">
            <h2 style="color:#003B73;">Welcome to the Smart Course Recommender!</h2>
            <p style="font-size:17px; color:#333;">
                This intelligent system helps you discover the best courses based on your input using machine learning and fuzzy keyword search.
            </p>
            <ul style="font-size:16px; color:#333;">
                <li>🔍 Type a course name or related keywords</li>
                <li>📚 Get the top 5 recommended courses</li>
                <li>🌐 Visit the course links directly</li>
            </ul>
            <p style="font-size:17px; color:#666;"><i>Start exploring and upskill smartly!</i></p>
        </div>
        """, unsafe_allow_html=True)

    # ----------- RECOMMEND TAB -----------
    with tab2:
        st.markdown("<h3 style='color:#003B73;'>🔍 Search & Get Course Recommendations</h3>", unsafe_allow_html=True)
        user_input = st.text_input("Enter a course name or keywords", help="Example: 'data science', 'ai', etc.")
        if st.button("Find Recommendations"):
            if user_input:
                closest_course = get_closest_course_name(user_input, course_data)
                if closest_course:
                    st.success(f"Showing results for: **{closest_course.title()}**")
                    recommendations = recommend_courses(closest_course, course_data, similarity_matrix)
                    if recommendations:
                        display_course_cards(recommendations)
                    else:
                        st.warning("No similar courses found.")
                else:
                    st.warning("Couldn't find a matching course. Try different keywords.")
            else:
                st.warning("Please enter a keyword to search.")

    # ----------- ABOUT TAB -----------
    with tab3:
        st.markdown("""
        <div style="padding: 20px; background-color: #fef9ef; border-radius: 10px;">
            <h3 style="color:#6A0572;">About This Project</h3>
            <p style="font-size:16px; color:#444;">
                This project recommends online courses based on user preferences using:
            </p>
            <ul style="font-size:16px; color:#333;">
                <li>✅ Pre-trained similarity model using course content</li>
                <li>✅ Keyword-based fuzzy matching</li>
                <li>✅ Clean and modern UI using Streamlit</li>
            </ul>
            <p style="font-size:16px; color:#555;">Made with ❤️ by Himanshu Singh Bisht</p>
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
