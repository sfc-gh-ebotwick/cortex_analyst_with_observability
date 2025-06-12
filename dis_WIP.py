import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Union
import _snowflake
import pandas as pd
import streamlit as st
from snowflake.snowpark.context import get_active_session
from snowflake.snowpark.exceptions import SnowparkSQLException
from snowflake.cortex import complete
# Trulens imports for evaluation and feedback
from trulens.core import TruSession, Feedback, Select
from trulens.connectors.snowflake import SnowflakeConnector
from trulens.providers.cortex import Cortex
from trulens.apps.custom import instrument, TruCustomApp
# from trulens.apps.app import TruApp
import os

# Log Trulens version for debugging
# try:
#     import trulens
#     st.write(f"Trulens version: {trulens.__version__}")
# except ImportError:
#     st.error("Trulens not installed properly.")

# Enable Trulens OpenTelemetry tracing for debugging
# os.environ["TRULENS_OTEL_TRACING"] = "1"

# Configuration constants
AVAILABLE_SEMANTIC_MODELS_PATHS = [
    "CORTEX_ANALYST_DEMO.REVENUE_TIMESERIES.RAW_DATA/revenue_timeseries.yaml"
]
API_ENDPOINT = "/api/v2/cortex/analyst/message"
API_TIMEOUT = 50000  # in milliseconds
FEEDBACK_API_ENDPOINT = "/api/v2/cortex/analyst/feedback"
SUMMARY_MODELS = ["claude-3-5-sonnet"]
DATABASE = "CORTEX_ANALYST_DEMO"
SCHEMA = "REVENUE_TIMESERIES"
STAGE = "RAW_DATA"
FILE = "revenue_timeseries.yaml"

# Initialize Snowpark session for Snowflake queries
snowpark_session = get_active_session()

# --- Trulens Setup ---
# Custom connector to bypass database access issues in UDF
# class CustomSnowflakeConnector:
#     def __init__(self, snowpark_session, database, schema):
#         self.snowpark_session = snowpark_session
#         self.database = database
#         self.schema = schema
#     def get_app(self, app_id):
#         return None  # Minimal implementation
#     def __getattr__(self, name):
#         # Fallback to avoid attribute errors
#         return lambda *args, **kwargs: None

# Initialize SnowflakeConnector
try:
    conn = SnowflakeConnector(
        snowpark_session=snowpark_session,
        database=DATABASE,
        schema=SCHEMA
    )
    st.write(f"Connector type: {type(conn).__name__}")
except Exception as e:
    st.warning(f"SnowflakeConnector initialization failed: {str(e)}. Using custom connector.")
    # conn = CustomSnowflakeConnector(snowpark_session, DATABASE, SCHEMA)
    # st.write(f"Connector type: {type(conn).__name__}")

# Initialize TruSession
tru_session = TruSession()
tru_session.connector = conn
if tru_session.connector is None:
    st.error("Failed to set TruSession connector.")
else:
    st.write("TruSession connector initialized successfully.")
    st.write(snowpark_session)

# Initialize Cortex provider
provider = Cortex(snowpark_session, "llama3.1-70b")

# --- Trulens Feedback Definitions ---
final_answer_relevance = (
    Feedback(provider.relevance_with_cot_reasons, name="Final Answer Relevance")
    .on_input()
    .on_output()
)
# interpretation_accuracy = (
#     Feedback(provider.relevance_with_cot_reasons, name="Interpretation Accuracy")
#     .on(Select.RecordCalls.helper_function.rets['interpretation'])
#     .on(Select.RecordCalls.get_analyst_response.rets['message']['content'][0]['text'])
# )
# sql_relevance = (
#     Feedback(provider.relevance_with_cot_reasons, name="SQL Relevance")
#     .on(Select.RecordCalls.helper_function.rets['sql_gen'])
#     .on(Select.RecordCalls.get_analyst_response.rets['message']['content'][1]['statement'])
# )
# summarization_groundedness = (
#     Feedback(
#         provider.groundedness_measure_with_cot_reasons,
#         name="Summarization Groundedness"
#     )
#     .on(Select.RecordCalls.execute_query.rets)
#     .on_output()
# )
feedback_list = [final_answer_relevance]
#interpretation_accuracy, sql_relevance, summarization_groundedness


# --- CortexAnalyst Class for Instrumented Functions ---
class CortexAnalyst:
    @instrument
    def get_analyst_response(self, messages: List[Dict]) -> Tuple[Dict, Optional[str]]:
        """Calls Snowflake Cortex Analyst API to generate a response, serving as the main entry point for Trulens feedback."""
        request_body = {
            "messages": messages,
            "semantic_model_file": f"@{st.session_state.selected_semantic_model_path}",
        }
        resp = _snowflake.send_snow_api_request(
            "POST", API_ENDPOINT, {}, {}, request_body, None, API_TIMEOUT
        )
        parsed_content = json.loads(resp["content"])
        if resp["status"] < 400:
            return parsed_content, None
        error_msg = (
            f"🚨 Analyst API error\n\n"
            f"* response code: `{resp['status']}`\n"
            f"* request-id: `{parsed_content['request_id']}`\n"
            f"* error code: `{parsed_content['error_code']}`\n\n"
            f"Message:\n```\n{parsed_content['message']}\n```"
        )

        st.write(error_msg)
        return parsed_content, error_msg

    @instrument
    def execute_query(self, query: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
        """Executes a SQL query and returns the result as a DataFrame."""
        try:
            df = snowpark_session.sql(query).to_pandas()
            return df, None
        except SnowparkSQLException as e:
            return None, str(e)

    @instrument
    def summarize_results(self, df: pd.DataFrame, user_prompt: str) -> str:
        """Generates a natural language summary of query results using Cortex."""
        if df.empty:
            data_str = 'No data was returned by the query'
        else:
            max_rows = 40
            df_limited = df.head(max_rows)
            data_str = df_limited.to_csv(index=False, lineterminator='\n')
            if len(df) > max_rows:
                data_str += f"\n(Note: Showing first {max_rows} rows of {len(df)} total rows)"
        
        sanitized_prompt = user_prompt[:1000]
        prompt = (
            "Based on the user's question: '{}', "
            "here is a sample of the query results in CSV format:\n{}"
            "\nProvide a concise natural language summary of the data in context of the question. Format results neatly with markdown"
        ).format(sanitized_prompt, data_str)
        
        summary = complete('claude-3-5-sonnet', prompt)
        
        
        # snowpark_session.sql(
        #     "SELECT SNOWFLAKE.CORTEX.COMPLETE(?, ?) AS summary",
        #     params=["claude-3-5-sonnet", prompt]
        # ).to_pandas()
        
        return summary

    @instrument
    def helper_function(self, prompt: str) -> dict:
        """Generates inputs for Trulens feedback evaluation."""
        return {
            'interpretation': f"Interpret or clarify the following prompt: {prompt}",
            'sql_gen': f"Create sql that would be appropriate to answer the input prompt - {prompt}"
        }
    @instrument
    def process_user_input(self, prompt: str):
        """Processes user input, generates response, and handles SQL execution and summarization."""
        st.session_state.warnings = []
        new_user_message = {"role": "user", "content": [{"type": "text", "text": prompt}]}
        st.session_state.messages.append(new_user_message)
        with st.chat_message("user"):
            user_msg_index = len(st.session_state.messages) - 1
            display_message(new_user_message["content"], user_msg_index)
    
        with st.chat_message("analyst"):
            with st.spinner("Waiting for Infrastructure Analyst's response..."):
                time.sleep(1)
                response, error_msg = self.get_analyst_response(st.session_state.messages)
                analyst_message = {
                    "role": "analyst",
                    "content": response["message"]["content"] if error_msg is None else [{"type": "text", "text": error_msg}],
                    "request_id": response["request_id"],
                }
                if "warnings" in response:
                    st.session_state.warnings = response["warnings"]
                if error_msg:
                    st.session_state["fire_API_error_notify"] = True
                # if analyst_message["content"][1]["type"] != "sql":
                #     st.session_state.messages.append(analyst_message)
                #     st.write(analyst_message["content"][0]["text"])
                #     st.rerun()
                else:
                    with st.spinner("Running query and summarizing results..."):
                        time.sleep(1)
                        sql = analyst_message["content"][1]["statement"]
                        df, error = self.execute_query(sql)
                        if error:
                            st.error(f"Query failed: {str(error)}")
                        else:
                            user_prompt = next((msg["content"][0]["text"] for msg in reversed(st.session_state.messages) if msg["role"] == "user"), "Unknown query")
                            summary = self.summarize_results(df, user_prompt)
                            analyst_message["content"].append({"type": "text", "text": summary})
                            st.session_state.messages.append(analyst_message)
                            if feedback_enabled:
                                st.write("FEEDBACK IS ENABLED")
                                try:
                                    with st.sidebar.expander("Trulens Feedback", expanded=True):
                                        records = tru_session.get_records_and_feedback(app_ids=["CORTEX_ANALYST"])[0]
                                        if not records.empty:
                                            for feedback_name in [f.name for f in feedback_list]:
                                                score = records[records["feedback_name"] == feedback_name]["score"].iloc[-1] if feedback_name in records["feedback_name"].values else "N/A"
                                                st.write(f"{feedback_name}: {score}")
                                except Exception as e:
                                    st.sidebar.warning(f"Failed to display feedback: {str(e)}")
                            st.rerun()
        return summary


# Instantiate CortexAnalyst and TruCustomApp
CA = CortexAnalyst()
feedback_enabled = True
try:
    # Attempt with feedback logging
    tru_app = TruCustomApp(
        app = CA,
        app_id="CORTEX_ANALYST",
        app_name = "Storage-Analyst_with_Trulens",
        app_version= 'v3',
        feedbacks=feedback_list,
        session=tru_session
    )
    
    feedback_enabled = True
    st.write("TruCustomApp initialized with feedback logging.")
    st.write(f"TruCustomApp connector set: {tru_app.connector is not None}")
except Exception as e:
    st.warning(f"Failed to initialize TruCustomApp with feedback: {str(e)}")
    # try:
    #     # Fallback without feedback logging
    #     tru_app = TruCustomApp(
    #         CA,
    #         app_id="CORTEX_ANALYST",
    #         app_version="Storage-Analyst_with_Trulens",
    #         main_method=CA.get_analyst_response,
    #         tru=tru_session
    #     )
    #     #tru_app.connector = conn
    #     st.write("TruCustomApp initialized without feedback logging.")
    # except Exception as e2:
    #     st.error(f"Fallback initialization failed: {str(e2)}")
    #     raise

def main():
    """Main function to initialize and run the Streamlit app."""
    if "messages" not in st.session_state:
        reset_session_state()
    show_header_and_sidebar()
    display_conversation()
    # handle_user_inputs()
    
    user_input = st.chat_input("What is your question?")
    st.write(f"User Input: {user_input}")
    if user_input:
        st.write("FOUND USER INPUT!")

        with tru_app as recording:
            CA.process_user_input(user_input)
        st.write("RECORDED_APP_CALL!")
        
    handle_error_notifications()

def reset_session_state():
    """Resets Streamlit session state variables."""
    st.session_state.messages = []
    st.session_state.active_suggestion = None
    st.session_state.warnings = []
    st.session_state.form_submitted = {}

def show_header_and_sidebar():
    """Displays the app header and sidebar with model selection and reset button."""
    st.logo("https://www.disneystudios.com/fonts/logo.svg")
    st.title("Infrastructure Analyst: Storage Analytics")
    st.markdown("""
        This app's semantic data model contains information about storage capacity, 
        usage and costs across different storage clusters and locations. 
        <ul>
          <li>You can ask about storage metrics like raw and usable capacity in different units (TB, TiB, PB)</li>
          <li>Track costs on a monthly/yearly basis, and break down storage usage by different banners and shows</li>
          <li>The data allows you to understand storage utilization patterns, costs, and capacity planning needs across different geographic locations and business units</li>
        </ul>
    """, unsafe_allow_html=True)
    with st.sidebar:
        st.selectbox(
            "Selected semantic model:",
            AVAILABLE_SEMANTIC_MODELS_PATHS,
            format_func=lambda s: s.split("/")[-1],
            key="selected_semantic_model_path",
            on_change=reset_session_state,
        )
        st.divider()
        _, btn_container, _ = st.columns([2, 6, 2])
        if btn_container.button("Clear Chat History", use_container_width=True):
            reset_session_state()
        with st.expander("Trulens Feedback", expanded=False):
            if feedback_enabled:
                st.write("Feedback logging enabled. Scores will appear after queries.")
            else:
                st.write("Feedback logging disabled due to connector issues.")

def handle_user_inputs():
    """Handles user input from the chat interface."""
    user_input = st.chat_input("What is your question?")
    if user_input:
        with tru_app as recording:
            CA.process_user_input(user_input)
        st.write("TRULENS_FINISHED_RECORDING")
        st.session_state.active_suggestion = None
    elif st.session_state.active_suggestion:
        suggestion = st.session_state.active_suggestion
        st.session_state.active_suggestion = None
        with tru_app as recording:
            CA.process_user_input(suggestion)

def handle_error_notifications():
    """Displays error notifications for API failures."""
    if st.session_state.get("fire_API_error_notify"):
        st.toast("An API error has occurred!", icon="🚨")
        st.session_state["fire_API_error_notify"] = False


def display_conversation():
    """Displays the chat conversation history."""
    st.session_state.active_suggestion = None
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            if message["role"] == "analyst":
                display_message(message["content"], idx, message["request_id"])
            else:
                display_message(message["content"], idx)

def display_message(content: List[Dict], message_index: int, request_id: Optional[str] = None):
    """Displays a single message's content (text, suggestions, or SQL)."""
    for item in content:
        if item["type"] == "text":
            st.markdown(item["text"])
        elif item["type"] == "suggestions":
            for suggestion_index, suggestion in enumerate(item["suggestions"]):
                if st.button(suggestion, key=f"suggestion_{message_index}_{suggestion_index}"):
                    st.session_state.active_suggestion = suggestion
        elif item["type"] == "sql":
            display_sql_query(item["statement"], message_index, item["confidence"], request_id)

def display_sql_query(sql: str, message_index: int, confidence: dict, request_id: Optional[str]):
    """Displays SQL query, results, and charts."""
    with st.expander("SQL Query", expanded=False):
        st.code(sql, language="sql")
        display_sql_confidence(confidence)
    with st.expander("Results", expanded=True):
        with st.spinner("Running SQL..."):
            df, error = CA.execute_query(sql)
            if df is None:
                st.error(f"Could not execute SQL query. Error: {str(error)}")
            elif df.empty:
                st.write("Query returned no data")
            else:
                data_tab, chart_tab = st.tabs(["Data 📄", "Chart 📉"])
                with data_tab:
                    st.dataframe(df, use_container_width=True)
                with chart_tab:
                    display_charts_tab(df, message_index)
    if request_id:
        display_feedback_section(request_id)

def display_sql_confidence(confidence: dict):
    """Displays confidence details for verified SQL queries."""
    if not confidence:
        return
    verified_query_used = confidence.get("verified_query_used")
    with st.popover("Verified Query Used"):
        if not verified_query_used:
            st.text("No verified query used to generate this SQL.")
            return
        st.text(f"Name: {verified_query_used['name']}")
        st.text(f"Question: {verified_query_used['question']}")
        st.text(f"Verified by: {verified_query_used['verified_by']}")
        st.text(f"Verified at: {datetime.fromtimestamp(verified_query_used['verified_at'])}")
        st.text("SQL query:")
        st.code(verified_query_used["sql"], language="sql", wrap_lines=True)

def display_charts_tab(df: pd.DataFrame, message_index: int):
    """Displays interactive charts for query results."""
    if len(df.columns) < 2:
        st.write("At least 2 columns are required for charts.")
        return
    all_cols_set = set(df.columns)
    col1, col2 = st.columns(2)
    x_col = col1.selectbox("X axis", all_cols_set, key=f"x_col_select_{message_index}")
    y_col = col2.selectbox("Y axis", all_cols_set.difference({x_col}), key=f"y_col_select_{message_index}")
    chart_type = st.selectbox(
        "Select chart type",
        options=["Line Chart 📈", "Bar Chart 📊"],
        key=f"chart_type_{message_index}",
    )
    if chart_type == "Line Chart 📈":
        st.line_chart(df.set_index(x_col)[y_col])
    elif chart_type == "Bar Chart 📊":
        st.bar_chart(df.set_index(x_col)[y_col])

def display_feedback_section(request_id: str):
    """Displays a feedback form for SQL query evaluation."""
    with st.popover("📝 Query Feedback"):
        if request_id not in st.session_state.form_submitted:
            with st.form(f"feedback_form_{request_id}", clear_on_submit=True):
                positive = st.radio("Rate the generated SQL", options=["👍", "👎"], horizontal=True)
                positive = positive == "👍"
                feedback_message = st.text_input("Optional feedback message")
                submitted = st.form_submit_button("Submit")
                if submitted:
                    err_msg = submit_feedback(request_id, positive, feedback_message)
                    st.session_state.form_submitted[request_id] = {"error": err_msg}
                    st.rerun()
        elif st.session_state.form_submitted[request_id]["error"] is None:
            st.success("Feedback submitted", icon="✅")
        else:
            st.error(st.session_state.form_submitted[request_id]["error"])

def submit_feedback(request_id: str, positive: bool, feedback_message: str) -> Optional[str]:
    """Submits user feedback for a query to the Cortex Analyst API."""
    request_body = {
        "request_id": request_id,
        "positive": positive,
        "feedback_message": feedback_message,
    }
    resp = _snowflake.send_snow_api_request(
        "POST", FEEDBACK_API_ENDPOINT, {}, {}, request_body, None, API_TIMEOUT
    )
    if resp["status"] == 200:
        return None
    parsed_content = json.loads(resp["content"])
    return (
        f"🚨 Feedback API error\n\n"
        f"* response code: `{resp['status']}`\n"
        f"* request-id: `{parsed_content['request_id']}`\n"
        f"* error code: `{parsed_content['error_code']}`\n\n"
        f"Message:\n```\n{parsed_content['message']}\n```"
    )

if __name__ == "__main__":
    main()
