from typing import Dict, List, Optional

import _snowflake
import json
import streamlit as st
import time
from functools import partial
from snowflake.snowpark.context import get_active_session
from snowflake.cortex import complete
from trulens.core import TruSession
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.apps.custom import instrument
from trulens.providers.cortex.provider import Cortex
from trulens.core import Feedback, SnowflakeFeedback, Select
from trulens.apps.custom import TruCustomApp


snowpark_session = get_active_session()

DATABASE = "A_NFL_RAG"
SCHEMA = "DATA"
STAGE = "SEMANTIC"
FILE = "weekly_player_stats_2024.yaml"

try:
    tru_session 
    st.write("Using existing TruSession")
except:
    @st.cache_resource
    def create_tru_session():
        conn = SnowflakeConnector(snowpark_session=snowpark_session)
        tru_session = TruSession(connector=conn)
        #Define llm for eval
        provider = Cortex(snowpark_session.connection, "llama3.1-70b")
        return  tru_session, provider
    tru_session, provider = create_tru_session()


# How well does the final summarization answer the users initial prompt?
final_answer_relevance = (
            Feedback(provider.relevance_with_cot_reasons, name = "Final Answer relevance")
            .on_input()
            .on_output())
    
# How accurately is cortex analyst inpreting the users prompt?
interpretation_accuracy = (
                Feedback(provider.relevance_with_cot_reasons,
                name="Interpretation Accuracy")
                .on(Select.RecordCalls.helper_function.rets['interpretation'])
                .on(Select.RecordCalls.call_analyst_api.rets['message']['content'][0]['text']))

# How relevant is the generated SQL to the users prompt?
sql_relevance = (
            Feedback(provider.relevance_with_cot_reasons, 
            name = "SQL relevance")
            .on(Select.RecordCalls.helper_function.rets['sql_gen'])
            .on(Select.RecordCalls.call_analyst_api.rets['message']['content'][1]['statement']))


# How well grounded in the sql results is the summarization
summarization_groundedness = (Feedback(
                partial(provider.groundedness_measure_with_cot_reasons, 
                use_sent_tokenize=False), 
                name="Summarization Groundedness", 
                use_sent_tokenize=False)
                .on(Select.RecordCalls.process_sql.rets)
                .on_output())


feedback_list = [interpretation_accuracy, sql_relevance, summarization_groundedness, final_answer_relevance]


class CortexAnalyst():
    @instrument
    def call_analyst_api(self,prompt: str) -> dict:

        """Calls the REST API and returns the response."""
        request_body = {
            "messages": st.session_state.messages,
            "semantic_model_file": f"@{DATABASE}.{SCHEMA}.{STAGE}/{FILE}",
        }
        resp = _snowflake.send_snow_api_request(
            "POST",
            f"/api/v2/cortex/analyst/message",
            {},
            {},
            request_body,
            {},
            30000,
        )
        if resp["status"] < 400:
            return json.loads(resp["content"])
        else:
            st.session_state.messages.pop()
            raise Exception(
                f"Failed request with status {resp['status']}: {resp}"
            )

    @instrument
    def process_api_response(self, prompt: str) -> str:
        """Processes a message and adds the response to the chat."""
        st.session_state.messages.append(
            {"role": "user", "content": [{"type": "text", "text": prompt}]}
        )
        with st.chat_message("user"):
            st.markdown(prompt)
        with st.chat_message("assistant"):
            with st.spinner("Generating response..."):
                # response = "who had the most rec yards week 10"
                response = self.call_analyst_api(prompt=prompt)
                request_id = response["request_id"]
                content = response["message"]["content"]
                st.session_state.messages.append(
                    {**response['message'], "request_id": request_id}
                )
                final_return = self.process_sql(content=content, request_id=request_id)  # type: ignore[arg-type]
                
        return final_return
        
    @instrument
    def process_sql(self,
        content: List[Dict[str, str]],
        request_id: Optional[str] = None,
        message_index: Optional[int] = None,
    ) -> str:
        """Displays a content item for a message."""
        message_index = message_index or len(st.session_state.messages)
        sql_markdown = 'No SQL returned!'
        if request_id:
            with st.expander("Request ID", expanded=False):
                st.markdown(request_id)
        for item in content:
            if item["type"] == "text":
                st.markdown(item["text"])
            elif item["type"] == "suggestions":
                with st.expander("Suggestions", expanded=True):
                    for suggestion_index, suggestion in enumerate(item["suggestions"]):
                        if st.button(suggestion, key=f"{message_index}_{suggestion_index}"):
                            st.session_state.active_suggestion = suggestion
            elif item["type"] == "sql":
                sql_markdown = self.execute_sql(sql = item["statement"])

        return sql_markdown

    # @st.cache_data
    @instrument
    def execute_sql(self, sql: str) -> None:
        with st.expander("SQL Query", expanded=False):
            st.code(sql, language="sql")
        with st.expander("Results", expanded=True):
            with st.spinner("Running SQL..."):
                session = get_active_session()
                df = session.sql(sql).to_pandas()
                if len(df.index) > 1:
                    data_tab, line_tab, bar_tab = st.tabs(
                        ["Data", "Line Chart", "Bar Chart"]
                    )
                    data_tab.dataframe(df)
                    if len(df.columns) > 1:
                        df = df.set_index(df.columns[0])
                    with line_tab:
                        st.line_chart(df)
                    with bar_tab:
                        st.bar_chart(df)
                else:
                    st.dataframe(df)

        return df.to_markdown(index=False)

    @instrument
    def summarize_sql_results(self, prompt: str) -> str:
        sql_result = self.process_api_response(prompt)
        st.write("Summarizing result...")
        summarized_result = complete("claude-3-5-sonnet", 
                                     f'''Summarize the following input prompt and corresponding SQL result 
                                     from markdown into a succint human readable summary. 
                                     Original prompt - {prompt}
                                     Sql result markdown - {sql_result}''')
        st.write(f"**{summarized_result}**")
        helper = self.helper_function(prompt)
        return summarized_result

    @instrument
    def helper_function(self, prompt:str) -> dict:
        helper_dict = {}
        helper_dict['interpretation'] = f"Interpret or clarify the following prompt: {prompt}"
        helper_dict['sql_gen'] = f"Create sql that would be appropriate to answer the input prompt - {prompt}"
        return helper_dict



#instantiate class
CA = CortexAnalyst()

# CREATE TRULENS APP WITH CA instance
tru_app = TruCustomApp(
    CA,
    app_id="CORTEX_ANALYST",
    app_version="sql_and_summarization_multi_agent_app",
    feedbacks=feedback_list
)


def show_conversation_history() -> None:
    for message_index, message in enumerate(st.session_state.messages):
        chat_role = "assistant" if message["role"] == "analyst" else "user"
        with st.chat_message(chat_role):
               try:
                   CA.process_sql(
                        content=message["content"],
                        request_id=message.get("request_id"),
                        message_index=message_index,
                    )
               except: 
                   st.write("No history found!")


def reset() -> None:
    st.session_state.messages = []
    st.session_state.suggestions = []
    st.session_state.active_suggestion = None



st.title(f":football: NFL Text to SQL Assistant with Snowflake Cortex :football:")

st.markdown(f"Semantic Model: `{FILE}`")

if "messages" not in st.session_state:
    reset()

with st.sidebar:
    if st.button("Reset conversation"):
        reset()

show_conversation_history()

if user_input := st.chat_input("What is your question?"):

    # Test the pipeline
    with tru_app as recording:
        CA.summarize_sql_results(prompt=user_input)
    
if st.session_state.active_suggestion:
    CA.process_api_response(prompt=st.session_state.active_suggestion)
    st.session_state.active_suggestion = None
