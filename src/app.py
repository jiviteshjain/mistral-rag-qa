import gc

import streamlit as st
from omegaconf import OmegaConf
import torch

from rag_qa import RagQA

CONFIG_PATH = "rag_pipeline/conf/inference.yaml"


@st.cache_resource(
    show_spinner="Loading models and indices. This might take a while. Go get hydrated..."
)
def get_rag_qa():
    gc.collect()
    torch.cuda.empty_cache()
    conf = OmegaConf.load(CONFIG_PATH)
    rag_qa = RagQA(conf)
    rag_qa.load()
    return rag_qa


left_column, cent_column, last_column = st.columns(3)
with cent_column:
    st.image("pittsburgh.webp", width=400)
st.title("Know anything about Pittsburgh")

# Initialize the RagQA model, might be already cached.
_ = get_rag_qa()

# Run QA
st.subheader("Ask away:")
question = st.text_input("Ask away:", "", label_visibility="collapsed")
submit = st.button("Submit")

st.markdown(
    """
    > **For example, ask things like:**
    >
    > Who is the largest employer in Pittsburgh?  
    > Where is the Smithsonian affiliated regional history museum in Pittsburgh?  
    > Who is the president of CMU?
    ---
    """,
    unsafe_allow_html=False,
)

if submit:
    if not question.strip():
        st.error("Machine Learning still can't read minds. Please enter a question.")
    else:
        try:
            with st.spinner("Combing through 20,000+ documents from 14,000+ URLs..."):
                answer, sources = get_rag_qa().answer(question)

            st.subheader("Answer:")
            st.write(answer)

            st.write("")

            with st.expander("Show Sources"):
                st.subheader("Sources:")
                for i, source in enumerate(sources):
                    st.markdown(f"**Name:** {source.name}")
                    st.markdown(f"**Index ID:** {source.index_id}")
                    st.markdown(f"**Text:**")
                    st.write(source.text)
                    if i < len(sources) - 1:
                        st.markdown("---")

        except Exception as e:
            st.error(f"An error occurred: {e}")
