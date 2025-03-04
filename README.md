Demo showing AI observability integration with Cortex Analyst in Snowflake


# This repo contains a sample streamlit app and enviornment.yaml file for running Cortex Analyst in Snowflake with AI Observability
## The streamlit app contains functionality for the following
 
 * ### Calling Cortex Analyst REST API to generate SQL for user prompt
 * ### Executing SQL Query on underlying snowflake data
 * ### Using a summarization agent to transcribe the sql results into a human readable text string

## AI Observability grants us the ability to do the following on top of this application
 * ### Traces each stage of the application to understand the input and output data and how long each stage takes to execute
 * ### Runs several feedback functions to determine the performance of various stages of the application
   * #### Interpretation Accuracy - How accurately is cortex analyst inpreting the users prompt?
   * #### SQL Relevance - How relevant is the generated SQL to the users prompt?
   * #### Summarization Groundedness - How well grounded in the sql results is the summarization?
   * #### Final Answer Relevance - How well does the final summarization answer the users initial prompt?
  
##### This demo is meant to serve as an illustrative example of how observability could be run on top of an agentic sql generation application
