import streamlit as st
#from transformers import pipeline
from bot import ChatBot


mybot=ChatBot()
workflow=mybot()

# Set up the Streamlit app UI
st.title("ChatBot with LangGraph")
st.write("Ask any question, and I'll try to answer it!")

# Input text box for the question
question = st.text_input("Enter your question here:")
input={"messages": [question]}

# Button to get the answer
# Button to get the answer
if st.button("Get Answer"):
    if input:
        response = workflow.invoke(input)
        last_message = response['messages'][-1]

        # Extract content safely
        if isinstance(last_message.content, list):
            # Join text parts if it's a list of chunks
            answer = " ".join([chunk.get("text", "") for chunk in last_message.content])
        else:
            answer = str(last_message.content)

        st.write("**Answer:**", answer)
    else:
        st.warning("Please enter a question to get an answer.")


# Additional styling (optional)
st.markdown("---")
st.caption("Powered by Streamlit and Transformers")

