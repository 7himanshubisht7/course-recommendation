import streamlit as st
import pickle
import pandas as pd

# File paths
COURSE_DATA_PATH = r"C:\Users\Acer\OneDrive\Desktop\course recommendation\Coursera.csv"
SIMILARITY_PATH = r"C:\Users\Acer\OneDrive\Desktop\course recommendation\similarity.pkl"

# Load CSV and Pickle Files
@st.cache_data
def load_data(course_data_path, similarity_path):
    try:
        # Load course data from CSV
        course_data = pd.read_csv(course_data_path)

        # Load similarity matrix from pickle file
        with open(similarity_path, 'rb') as file:
            similarity_matrix = pickle.load(file)
        
        return course_data, similarity_matrix
    except FileNotFoundError as e:
        st.error(f"File not found: {e}")
        return None, None
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None, None

# Recommendation Function
def recommend_courses(course, course_data, similarity_matrix):
    # Handle case-insensitive matching
    course = course.strip().lower()  # Remove leading/trailing spaces and make lowercase
    course_data['course_name_lower'] = course_data['Course Name'].str.lower()
    
    # Check if the course exists
    if course not in course_data['course_name_lower'].values:
        st.warning("Course not found. Please check the course name.")
        return None

    # Get the course index
    course_index = course_data[course_data['course_name_lower'] == course].index[0]
    distances = similarity_matrix[course_index]
    
    # Get the top 5 recommendations
    course_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]
    recommendations = []

    # Collect the recommended courses
    for i in course_list:
        recommended_course = course_data.iloc[i[0]]
        recommendations.append({
            'course_name': recommended_course['Course Name'],  # Accessing 'Course Name' directly
            'course_url': recommended_course['Course URL'],    # Accessing 'Course URL' directly
            'course_description': recommended_course['Course Description'],  # Description if needed
            'university': recommended_course['University']  # University if needed
        })
    return recommendations


# Streamlit App
def main():
    st.title("Course Recommendation System")

    # Sidebar menu
    menu = ["Home", "Recommend", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    # Load data
    course_data, similarity_matrix = load_data(COURSE_DATA_PATH, SIMILARITY_PATH)

    if course_data is None or similarity_matrix is None:
        st.error("Error loading the data files. Please check the paths and try again.")
        return

    if choice == "Home":
        st.subheader("Home")
        st.write("Welcome to the Course Recommendation System!")
        st.dataframe(course_data.head())

    elif choice == "Recommend":
        st.subheader("Recommend Courses")

        # Input from user
        course_name = st.text_input("Enter a course name:")
        if st.button("Get Recommendations"):
            if course_name:
                recommendations = recommend_courses(course_name, course_data, similarity_matrix)
                if recommendations:
                    st.write("Top Recommended Courses:")
                    for rec in recommendations:
                        st.markdown(
                            f"- **[{rec['course_name']}]({rec['course_url']})** by {rec['university']}<br>"
                            f"   *{rec['course_description']}*",
                            unsafe_allow_html=True
                        )
                else:
                    st.warning("No recommendations found.")
            else:
                st.warning("Please enter a course name.")

    elif choice == "About":
        st.subheader("About")
        st.write("This application provides course recommendations using a pre-trained model.")

if __name__ == '__main__':
    main()
