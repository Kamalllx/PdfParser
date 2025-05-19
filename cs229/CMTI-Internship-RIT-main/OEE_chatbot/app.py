import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import os
import calendar
from datetime import date
from typing import Optional, List
from pydantic import BaseModel, Field
import plotly.express as px

# LangChain imports
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.tools import tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables import RunnablePassthrough

# --- Configuration ---
# Ensure your GROQ_API_KEY is set as an environment variable
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

if not groq_api_key:
    st.error(
        "GROQ_API_KEY environment variable not set. Please set it and rerun the app."
    )
    st.stop()

DATA_FILE = "synthetic_machine_data.csv" # Ensure this matches your generated file name
# We'll still need the SHIFT_DURATION_MINUTES to potentially calculate Planned Time sums
SHIFT_DURATION_MINUTES = 8 * 60 # 8 hours per shift entry as planned time
# Assume an ideal cycle time for Performance calculation when recalculating aggregated OEE
IDEAL_CYCLE_TIME_SECONDS_APP = 30

# --- Helper Function for Time Conversion ---
def hhmmss_to_minutes(time_str):
    """Converts HH:MM:SS string to total minutes (float)."""
    if pd.isna(time_str) or not isinstance(time_str, str):
        return 0.0
    try:
        h, m, s = map(int, time_str.split(':'))
        return h * 60 + m + s / 60.0
    except ValueError:
        # st.warning(f"Could not parse time string: '{time_str}'") # Commented out to avoid spamming warnings
        return 0.0


# --- Data Loading (Cached) ---
@st.cache_data
def load_oee_data(file_path: str):
    """Loads synthetic machine data from a CSV file and prepares it."""
    if not os.path.exists(file_path):
        st.error(
            f"Data file '{file_path}' not found. Please run the data generation script."
        )
        return None
    try:
        df = pd.read_csv(file_path)
        
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        # Handle both 'updatedate' and 'update_date' from potential previous versions or typos
        if 'updatedate' in df.columns:
             df.rename(columns={'updatedate': 'update_date'}, inplace=True)
        if 'update_date' in df.columns:
             df["update_date"] = pd.to_datetime(df["update_date"])

        # Convert time duration strings to minutes
        df["off_time_minutes"] = df["off_time"].apply(hhmmss_to_minutes)
        df["idle_time_minutes"] = df["idle_time"].apply(hhmmss_to_minutes)
        df["production_time_minutes"] = df["production_time"].apply(hhmmss_to_minutes)

        # Add derived time columns for context, based on shift duration
        df["planned_time_minutes"] = SHIFT_DURATION_MINUTES # Planned time per shift entry
        df["running_time_minutes"] = df["production_time_minutes"] # Production time is the actual running time
        df["downtime_minutes"] = df["off_time_minutes"] + df["idle_time_minutes"]

        # Add Date/Time components for filtering and aggregation
        df["Date"] = df["timestamp"].dt.date
        df["Year"] = df["timestamp"].dt.year
        df["Month"] = df["timestamp"].dt.month
        # Need Week string for grouping to avoid ISO calendar year issues across year boundaries
        df["Week"] = df["timestamp"].dt.isocalendar().week # Keep as int for display if needed
        df["Year_Week"] = df["timestamp"].dt.strftime("%Y-W%W")
        df["Year_Month"] = df["timestamp"].dt.strftime("%Y-%m")


        # Convert machine_id to string for consistent filtering with multiselect
        df['machine_id'] = df['machine_id'].astype(str)

        st.success(
            f"Successfully loaded data from '{file_path}'. Data shape: {df.shape}"
        )
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


# --- Data Filtering ---
def filter_data(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
    selected_machines: List[str],
) -> pd.DataFrame:
    """Filters DataFrame based on date range and machine IDs."""
    df_filtered = df.copy()

    # Date filtering - using the 'Date' column (which is date objects)
    df_filtered = df_filtered[
        (df_filtered["Date"] >= start_date) & (df_filtered["Date"] <= end_date)
    ]

    # Machine filtering (using machine_id - string type)
    if selected_machines:
        df_filtered = df_filtered[df_filtered["machine_id"].isin(selected_machines)]

    return df_filtered

# --- Data Aggregation for Plotting Time Series ---
# This aggregates the BASE metrics over time and then RE-CALCULATES OEE
# This is the standard way to get statistically correct OEE for aggregated periods
def aggregate_oee_data_time_series(df_subset: pd.DataFrame, aggregation_level: str) -> pd.DataFrame:
    """Aggregates data by the specified time level and calculates OEE metrics
       from aggregated base metrics (Planned, Running, Good/Total) for time series plotting."""
    if df_subset.empty:
        return pd.DataFrame()

    if aggregation_level == "Daily":
        group_col = "Date"
        sort_order = "Date"
    elif aggregation_level == "Weekly":
        group_col = "Year_Week"
        sort_order = "Year_Week" # Sorts chronologically due to YYYY-WW format
    elif aggregation_level == "Monthly":
        group_col = "Year_Month"
        sort_order = "Year_Month" # Sorts chronologically due to YYYY-MM format
    elif aggregation_level == "Yearly": # Added Yearly aggregation
        group_col = "Year"
        sort_order = "Year" # Sorts chronologically by year integer
    else:
        raise ValueError(f"Invalid aggregation level: {aggregation_level}")

    # Aggregate base metrics (summing minutes and counts)
    agg_data = (
        df_subset.groupby(group_col)
        .agg(
            total_planned_time_minutes=("planned_time_minutes", "sum"),
            total_running_time_minutes=("running_time_minutes", "sum"), # Sum of production time
            total_downtime_minutes=("downtime_minutes", "sum"), # Sum of off + idle time
            total_units_produced=("total_parts", "sum"),
            total_good_units_produced=("good_parts", "sum"),
            # Also average the pre-calculated OEE metrics from raw data for comparison
            avg_availability_raw=("availability", "mean"),
            avg_performance_raw=("performance", "mean"),
            avg_quality_raw=("quality", "mean"),
            avg_oee_raw=("oee", "mean"),
        )
        .reset_index()
    )

    # Calculate OEE components from the aggregated base metrics
    # This is the statistically correct way to get OEE for the aggregated period
    # Availability = Total Running Time / Total Planned Time
    agg_data["Availability (%)"] = (
        agg_data["total_running_time_minutes"] / agg_data["total_planned_time_minutes"]
    ) * 100.0
    agg_data["Availability (%)"] = (
        agg_data["Availability (%)"].fillna(0).replace([np.inf, -np.inf], 0).clip(upper=100.0)
    )

    # Performance = (Ideal Cycle Time * Total Count) / Total Running Time
    ideal_run_time_minutes_agg = (
        agg_data["total_units_produced"] * (IDEAL_CYCLE_TIME_SECONDS_APP / 60.0)
    )
    agg_data["Performance (%)"] = (
        ideal_run_time_minutes_agg / agg_data["total_running_time_minutes"]
    ) * 100.0
    agg_data["Performance (%)"] = (
        agg_data["Performance (%)"].fillna(0).replace([np.inf, -np.inf], 0).clip(upper=100.0)
    )


    # Quality = Total Good Units / Total Units Produced
    agg_data["Quality (%)"] = (
        agg_data["total_good_units_produced"] / agg_data["total_units_produced"]
    ) * 100.0
    agg_data["Quality (%)"] = (
        agg_data["Quality (%)"].fillna(0).replace([np.inf, -np.inf], 0).clip(upper=100.0)
    )

    # OEE = Availability * Performance * Quality (from the re-calculated values)
    agg_data["OEE (%)"] = (
        (agg_data["Availability (%)"] / 100.0)
        * (agg_data["Performance (%)"] / 100.0)
        * (agg_data["Quality (%)"] / 100.0)
        * 100.0
    )
    agg_data["OEE (%)"] = agg_data["OEE (%)"].clip(upper=100.0)

    # Convert relevant time columns to hours for clearer display in tables
    agg_data["total_planned_time_hrs"] = agg_data["total_planned_time_minutes"] / 60.0
    agg_data["total_running_time_hrs"] = agg_data["total_running_time_minutes"] / 60.0
    agg_data["total_downtime_hrs"] = agg_data["total_downtime_minutes"] / 60.0

    # Sort by the time column (handled by setting sort_order)
    agg_data = agg_data.sort_values(by=sort_order)

    return agg_data


# --- Calculation for Overall Filtered Period & Chatbot Query ---
# This calculates OEE for a specific subset (defined by sidebar filters or chat filters)
# It recalculates from summed base metrics for statistical correctness over the subset
def calculate_overall_oee_for_period(df_subset: pd.DataFrame) -> dict:
    """
    Calculates OEE and its components (Availability, Performance, Quality)
    from a DataFrame subset by aggregating base metrics within the subset
    and recalculating OEE.
    Handles potential division by zero by returning 0 for components.
    """
    if df_subset.empty:
        return {"error": "No data available for the selected filters."}

    # Aggregate key metrics over the subset
    total_planned_time_minutes = df_subset["planned_time_minutes"].sum()
    total_running_time_minutes = df_subset["running_time_minutes"].sum()
    total_downtime_minutes = df_subset["downtime_minutes"].sum()
    total_units_produced = df_subset["total_parts"].sum()
    total_good_units_produced = df_subset["good_parts"].sum()

    # Recalculate Availability, Performance, Quality, OEE from summed base metrics
    # Use the constant defined globally
    IDEAL_CYCLE_TIME_SECONDS_APP_FOR_CALC = IDEAL_CYCLE_TIME_SECONDS_APP # Use the defined constant


    # Availability
    availability = 0.0
    if total_planned_time_minutes > 0:
        availability = (total_running_time_minutes / total_planned_time_minutes) * 100.0

    # Performance
    ideal_run_time_minutes_subset = 0.0
    # Only calculate ideal run time if cycle time is positive
    if IDEAL_CYCLE_TIME_SECONDS_APP_FOR_CALC > 0:
         ideal_run_time_minutes_subset = (
             total_units_produced * (IDEAL_CYCLE_TIME_SECONDS_APP_FOR_CALC / 60.0)
         )

    performance = 0.0
    if total_running_time_minutes > 0:
        performance = (ideal_run_time_minutes_subset / total_running_time_minutes) * 100.0
    performance = min(performance, 100.0) # Performance can't exceed 100%

    # Quality
    quality = 0.0
    if total_units_produced > 0:
        quality = (total_good_units_produced / total_units_produced) * 100.0

    # OEE
    oee = (availability / 100.0) * (performance / 100.0) * (quality / 100.0) * 100.0
    oee = min(oee, 100.0) # OEE can't exceed 100%

    return {
        "availability": round(availability, 2),
        "performance": round(performance, 2),
        "quality": round(quality, 2),
        "oee": round(oee, 2),
        "total_planned_time_minutes": round(total_planned_time_minutes, 2),
        "total_running_time_minutes": round(total_running_time_minutes, 2),
        "total_downtime_minutes": round(total_downtime_minutes, 2),
        "total_units_produced": int(total_units_produced),
        "total_good_units_produced": int(total_good_units_produced),
        "assumed_ideal_cycle_time_sec": IDEAL_CYCLE_TIME_SECONDS_APP_FOR_CALC, # Include assumption
    }


# --- Conversational AI Tool Definition ---

class OEEQueryInput(BaseModel):
    machine_id: Optional[str] = Field(
        None,
        description="Specific Machine ID to filter by (e.g. '1', '2'). If not provided, data for all machines matching other filters will be used.",
    )
    month: Optional[str] = Field(
        None, description="Month to filter by (e.g., 'January', 'Feb', '03')."
    )
    year: Optional[int] = Field(
        None,
        description="Year to filter by (e.g., 2024, 2025). Must be a 4-digit year.",
    )

@tool("calculate_oee_chat_query", args_schema=OEEQueryInput)
def calculate_oee_chat_query_tool(
    machine_id: Optional[str] = None,
    month: Optional[str] = None,
    year: Optional[int] = None,
) -> str:
    """
    Calculates the Overall Equipment Efficiency (OEE) and its components
    (Availability, Performance, Quality) for machines based on filters
    extracted from a user's chat query. This tool processes explicit requests
    from the chat interface, independent of the UI filters set for plotting.

    Filters can include a specific Machine ID, Month (by name or number), and Year.
    If no filters are provided in the query, calculates OEE for the entire dataset.
    Filters are combined using AND logic.

    Provide the Machine ID (e.g. '1', '2'), Month (e.g., 'January', 'March', '6'),
    and Year (e.g., 2024) based on the user's request in the chat.
    Month can be the full name, abbreviation, or number (1-12).
    Year must be a 4-digit number.

    The OEE and component percentages are calculated by aggregating the raw data
    (total planned time, total running time, total units produced, total good units)
    for the filtered period and recalculating the percentages.
    Note: Performance calculation requires an assumed ideal cycle time (currently 30 seconds per part).

    Example calls:
    calculate_oee_chat_query(machine_id='1', month='January', year=2024)
    calculate_oee_chat_query(year=2025)
    calculate_oee_chat_query(month='July', year=2024)
    calculate_oee_chat_query(machine_id='5') # Calculates OEE for machine 5 across all time
    calculate_oee_chat_query() # Calculates overall OEE
    """
    # Access the globally loaded DataFrame (cached by Streamlit)
    df_filtered = st.session_state.oee_df.copy()

    # Apply filters from the chat query parameters
    if machine_id:
        df_filtered = df_filtered[df_filtered["machine_id"] == str(machine_id)]
        if df_filtered.empty:
            return f"No data found for Machine ID '{machine_id}'."

    if month:
        try:
            if month.isdigit():
                month_num = int(month)
                if not 1 <= month_num <= 12:
                    return f"Invalid month number provided in query: {month}. Please provide a number between 1 and 12."
            else:
                month_abbr_to_num = {
                    name.lower(): num
                    for num, name in enumerate(calendar.month_abbr)
                    if num > 0
                }
                month_name_to_num = {
                    name.lower(): num
                    for num, name in enumerate(calendar.month_name)
                    if num > 0
                }
                month_lower = month.lower()
                if month_lower in month_name_to_num:
                    month_num = month_name_to_num[month_lower]
                elif month_lower in month_abbr_to_num:
                    month_num = month_abbr_to_num[month_lower]
                else:
                    return f"Could not understand the month '{month}' from the query. Please use a full name (e.g., January), abbreviation (e.g., Jan), or number (e.g., 1)."

            df_filtered = df_filtered[df_filtered["Month"] == month_num]
            if df_filtered.empty:
                return f"No data found for the selected month ({month}) matching other criteria."

        except ValueError:
            return f"Could not parse the month '{month}' from the query. Please use a valid month name or number."

    if year:
        if (
            not isinstance(year, int) or year < 1000 or year > 3000
        ):
            return f"Invalid year provided in query: {year}. Please provide a 4-digit year."
        df_filtered = df_filtered[df_filtered["Year"] == year]
        if df_filtered.empty:
            return f"No data found for the year {year} matching other criteria."

    # Calculate OEE on the filtered data using the overall calculation function
    oee_results = calculate_overall_oee_for_period(df_filtered) # Use the overall calculation function

    if "error" in oee_results:
        return oee_results["error"]

    # Format the response string for the LLM to present
    filter_summary = []
    if machine_id:
        filter_summary.append(f"Machine ID '{machine_id}'")
    if month:
        filter_summary.append(f"Month '{month}'")
    if year:
        filter_summary.append(f"Year {year}")

    if not filter_summary:
        filter_text = "the entire dataset"
    else:
        filter_text = "for " + ", ".join(filter_summary)

    # Use the recalculated values and base sums in the response
    response_string = f"""
Calculation successful {filter_text}.
Overall Equipment Efficiency (OEE): {oee_results['oee']:.2f}%
Availability: {oee_results['availability']:.2f}%
Performance: {oee_results['performance']:.2f}% (recalculated using aggregated data and assuming ideal cycle time of {oee_results['assumed_ideal_cycle_time_sec']} sec/part)
Quality: {oee_results['quality']:.2f}%
Details (Aggregated): Total Planned Time={oee_results['total_planned_time_minutes']:.2f} min, Total Running Time={oee_results['total_running_time_minutes']:.2f} min, Total Downtime={oee_results['total_downtime_minutes']:.2f} min, Total Units Produced={oee_results['total_units_produced']}, Total Good Units Produced={oee_results['total_good_units_produced']}.
"""
    return response_string.strip()


# --- LangChain Agent Setup (Cached) ---


@st.cache_resource
def initialize_agent(oee_df: pd.DataFrame):
    """ Initializes the LangChain agent. """
    st.spinner("Initializing AI agent...")

    available_machines_list = sorted(oee_df["machine_id"].unique().tolist())
    available_years_list = sorted(oee_df["Year"].unique().tolist())
    available_years_range = (
        f"{min(available_years_list)} to {max(available_years_list)}"
        if available_years_list
        else "No data years available."
    )

    tools = [calculate_oee_chat_query_tool]

    system_prompt = f"""You are an AI assistant specialized in analyzing manufacturing data, specifically Overall Equipment Efficiency (OEE) for production machines.
The user is interacting with a Streamlit application that has two main sections:
1.  **Data Exploration & Plotting:** Where they can set filters using sidebar widgets (date range, machine ID) and view plots based on the filtered data. This tab includes overall metrics and time series charts based on the sidebar filters.
2.  **Conversational AI:** This chat interface, where you operate.

Your role in this chat interface is to answer specific questions about OEE based on explicit filters mentioned in the *user's chat query*. You use the `calculate_oee_chat_query` tool for this. The calculations you perform using the tool aggregate the raw data for the requested period and then recalculate the OEE components (Availability, Performance, Quality) from these aggregated sums. This is the standard way to calculate OEE for a period.
Note: The Performance calculation requires an assumed ideal cycle time (currently {IDEAL_CYCLE_TIME_SECONDS_APP} seconds per part) as this value is not directly available in the dataset for recalculation.

When a user asks for OEE in the chat, carefully extract the requested Machine ID, Month, and Year *from their message*.
Then, use the `calculate_oee_chat_query` tool with the extracted parameters.
If the user doesn't specify a filter in the chat message, use the tool without that filter.
If no filters are specified in the chat message at all, calculate OEE for the entire dataset using the tool.

After calling the tool, the tool will return a structured string with the calculation results or an error message.
Based on the tool's output, present the results in a clear, conversational manner, summarizing the calculated OEE, Availability, Performance, and Quality percentages, and mentioning the ideal cycle time assumption used for performance.
If the tool's output indicates an error or no data, inform the user appropriately.

You can also briefly mention that they can use the "Data Exploration & Plotting" tab to visualize trends over time using interactive filters, including daily, weekly, monthly, and **yearly** aggregations, and to see the overall OEE metrics for the selected period.

Available data covers the period from {available_years_range}.
Available Machine IDs: {', '.join(available_machines_list)}

Do not invent data or calculations. Only use the information returned by the `calculate_oee_chat_query` tool.
Maintain a helpful and professional tone.
"""

    llm = ChatGroq(model="meta-llama/llama-4-scout-17b-16e-instruct", temperature=0)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("placeholder", "{chat_history}"),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    # Setup chat history using Streamlit session state
    chat_history = ChatMessageHistory()
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            chat_history.add_user_message(msg["content"])
        else:
            chat_history.add_ai_message(msg["content"])

    agent = create_tool_calling_agent(llm, tools, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

    agent_with_chat_history = (
        RunnablePassthrough.assign(chat_history=lambda x: chat_history.messages)
        | agent_executor
    )

    return (
        agent_with_chat_history,
        chat_history,
    )


# --- Streamlit App UI ---

st.title("⚙️ Machine OEE Dashboard & Chatbot")

# --- Load data and initialize state ---
if "oee_df" not in st.session_state:
    st.session_state.oee_df = load_oee_data(DATA_FILE)
    if st.session_state.oee_df is None:
        st.stop()

# Initialize chat messages history
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": "Hello! I can help you analyze machine OEE data.",
        }
    ]


# Get available filter options from the data
available_machines = sorted(st.session_state.oee_df["machine_id"].unique().tolist())
min_date_data = st.session_state.oee_df["Date"].min()
max_date_data = st.session_state.oee_df["Date"].max()


# Initialize agent using the loaded data
agent_with_chat_history, chat_history_obj = initialize_agent(st.session_state.oee_df)


# --- Sidebar Filters ---
st.sidebar.header("Data Filters")

# --- Date Range Inputs (Separate) ---
# Get min/max dates from the loaded data for the picker bounds
min_date_data = st.session_state.oee_df["Date"].min()
max_date_data = st.session_state.oee_df["Date"].max()

# Use two separate date inputs for start and end dates
selected_start_date = st.sidebar.date_input(
    "Select Start Date",
    value=min_date_data, # Default to the earliest date in the data
    min_value=min_date_data,
    max_value=max_date_data,
    format="YYYY-MM-DD",
    key="sidebar_start_date" # Add key for uniqueness
)

selected_end_date = st.sidebar.date_input(
    "Select End Date",
    value=max_date_data, # Default to the latest date in the data
    min_value=min_date_data, # Allow selecting any date from min_date up to max_date
    max_value=max_date_data,
    format="YYYY-MM-DD",
    key="sidebar_end_date" # Add key for uniqueness
)

# Optional: Add a check if end date is before start date
if selected_start_date > selected_end_date:
    st.sidebar.warning("End date must be after or the same as the start date.")
    # Note: The filter_data function handles the case where start > end by returning an empty DataFrame.


# Machine Multiselect
selected_machines = st.sidebar.multiselect(
    "Select Machine(s)",
    options=available_machines,
    default=[],  # Default to show all machines if none selected
    key="sidebar_machines" # Add key
)

# --- Apply Filters and Aggregate for Display/Plotting ---
df_filtered_ui = filter_data(
    st.session_state.oee_df,
    selected_start_date,
    selected_end_date,
    selected_machines,
)

# --- Main Content Area (Tabs) ---
# Use st.tabs with a key to maintain state across reruns
# This replaces the old way of managing active_tab in session state
tab1, tab2 = st.tabs(["📊 Data Exploration & Plotting", "🤖 Conversational AI"])


with tab1: # Content for the first tab
    st.header("Data Exploration & Plotting")

    st.write("Use the filters in the sidebar to select the data you want to explore.")

    if df_filtered_ui.empty:
        st.warning("No data matches the selected filters.")
    else:
        st.write(f"Showing data for {len(df_filtered_ui)} rows matching filters.")

        # --- Display Time Series Plot ---
        st.subheader("Metric Trends Over Time")

        # Select Metric to Plot
        metric_options = [
             "OEE (%)", # This will be the re-calculated one from aggregation
             "Availability (%)",
             "Performance (%)",
             "Quality (%)",
             "total_units_produced",
             "total_good_units_produced",
             "total_planned_time_minutes",
             "total_running_time_minutes",
             "total_downtime_minutes",
             # Can optionally add raw averages for comparison in tables
             "avg_availability_raw", "avg_performance_raw", "avg_quality_raw", "avg_oee_raw"
        ]
        # Aggregate a small subset to get column names from the output for validation
        # Ensure df_filtered_ui is not empty before calling head()
        df_aggregated_temp = aggregate_oee_data_time_series(df_filtered_ui.head(10), "Daily") if not df_filtered_ui.empty else pd.DataFrame()
        available_metrics = [m for m in metric_options if m in df_aggregated_temp.columns]

        # Ensure default OEE is available if possible, otherwise default to the first available metric
        default_metric_index = available_metrics.index("OEE (%)") if "OEE (%)" in available_metrics else (0 if available_metrics else -1)
        # Handle case where available_metrics might be empty
        if default_metric_index != -1:
             metric_to_plot = st.selectbox(
                 "Select Metric to Plot",
                 options=available_metrics,
                 index=default_metric_index,
                 key="metric_to_plot_select" # Add key
             )
        else:
             st.warning("No metrics available to plot.")
             metric_to_plot = None # Set to None to prevent plotting logic from running


        # Select Aggregation Level - Includes Yearly
        aggregation_level = st.radio(
            "Aggregate data by",
            options=["Daily", "Weekly", "Monthly", "Yearly"],
            index=2,  # Default to Monthly
            horizontal=True,
            key="aggregation_level_radio" # Add a key
        )

        # Only proceed with plotting if a metric was selected (i.e., available_metrics was not empty)
        if metric_to_plot:
            df_aggregated_ts = aggregate_oee_data_time_series(df_filtered_ui, aggregation_level)

            if df_aggregated_ts.empty or metric_to_plot not in df_aggregated_ts.columns:
                st.warning(
                    f"No aggregated data or '{metric_to_plot}' column available for plotting with current filters/aggregation."
                )
            else:
                # Determine x-axis column and potentially convert Date type
                if aggregation_level == "Daily":
                    x_axis_col = "Date"
                    df_aggregated_ts[x_axis_col] = pd.to_datetime(df_aggregated_ts[x_axis_col])
                elif aggregation_level == "Weekly":
                    x_axis_col = "Year_Week"
                elif aggregation_level == "Monthly":
                    x_axis_col = "Year_Month"
                elif aggregation_level == "Yearly": # Added Yearly case
                     x_axis_col = "Year"
                else:
                     x_axis_col = df_aggregated_ts.columns[0] # Fallback - should not happen


                ts_fig = px.line(
                    df_aggregated_ts,
                    x=x_axis_col,
                    y=metric_to_plot,
                    title=f"{metric_to_plot} Trend Over Time ({aggregation_level})",
                    labels={x_axis_col: aggregation_level},
                    markers=True,
                )

                ts_fig.update_layout(xaxis_title=aggregation_level, yaxis_title=metric_to_plot)
                # Rotate labels for crowded time series unless it's Yearly
                if aggregation_level != "Yearly":
                     ts_fig.update_xaxes(tickangle=45)
                else:
                     ts_fig.update_xaxes(tickangle=0) # No rotation for yearly

                # Add % suffix to y-axis for percentage metrics
                if metric_to_plot in ["OEE (%)", "Availability (%)", "Performance (%)", "Quality (%)", "avg_availability_raw", "avg_performance_raw", "avg_quality_raw", "avg_oee_raw"]:
                     ts_fig.update_yaxes(tickformat='.2f%') # Use tickformat for percentages

                st.plotly_chart(ts_fig, use_container_width=True)

        st.markdown("---") # Separator

        # --- Display Overall OEE for Filtered Period (Bar Chart) ---
        st.subheader("Overall OEE Components for Selected Period")

        overall_oee_results = calculate_overall_oee_for_period(df_filtered_ui)

        if "error" in overall_oee_results:
             st.warning(overall_oee_results["error"])
        else:
             # Create a DataFrame for the overall bar chart
             overall_plot_data = {
                 'Metric': ['Availability', 'Performance', 'Quality'],
                 'Value (%)': [overall_oee_results['availability'],
                               overall_oee_results['performance'],
                               overall_oee_results['quality']]
             }
             overall_plot_df = pd.DataFrame(overall_plot_data)

             # Generate plot title including overall OEE and assumption
             # Determine date range string for the title
             date_range_str = ""
             if selected_start_date == selected_end_date:
                 date_range_str = f" on {selected_start_date.strftime('%Y-%m-%d')}"
             elif selected_start_date and selected_end_date:
                 date_range_str = f" from {selected_start_date.strftime('%Y-%m-%d')} to {selected_end_date.strftime('%Y-%m-%d')}"

             # Determine machine filter string for the title
             machine_str = ""
             if selected_machines:
                 # Compare selected machines to all available machines
                 if set(selected_machines) == set(available_machines):
                      machine_str = "All Machines"
                 elif len(selected_machines) == 1:
                      machine_str = f"Machine {selected_machines[0]}"
                 else:
                      # Display up to 3 machine IDs, then say 'and x others'
                      if len(selected_machines) > 3:
                          machine_str = f"Machines {', '.join(selected_machines[:3])} and {len(selected_machines) - 3} others"
                      else:
                           machine_str = f"Machines {', '.join(selected_machines)}"
                 machine_str = f" for {machine_str}"
             else:
                 machine_str = " for All Machines"


             overall_plot_title = f"Overall OEE Component Summary{machine_str}{date_range_str}<br><sup>(Overall OEE: {overall_oee_results['oee']:.2f}%, Performance based on {overall_oee_results['assumed_ideal_cycle_time_sec']} sec/part)</sup>"

             overall_fig = px.bar(
                 overall_plot_df,
                 x='Metric',
                 y='Value (%)',
                 text='Value (%)', # Display values on bars
                 title=overall_plot_title,
                 color='Metric', # Different color for each bar
                 range_y=[0, 100] # Ensure y-axis goes up to 100%
             )

             # Customize text labels on bars
             overall_fig.update_traces(texttemplate='%{text:.2f}%', textposition='outside')
             overall_fig.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
             # Corrected line: use tickformat for percentage suffix
             overall_fig.update_yaxes(title='Percentage (%)', tickformat='.2f%') # Use tickformat for percentages

             st.plotly_chart(overall_fig, use_container_width=True)

             # Display overall key figures below the chart
             st.markdown(f"""
             **Overall Metrics for Selected Data:**
             *   OEE: **{overall_oee_results['oee']:.2f}%**
             *   Availability: **{overall_oee_results['availability']:.2f}%**
             *   Performance: **{overall_oee_results['performance']:.2f}%**
             *   Quality: **{overall_oee_results['quality']:.2f}%**
             *   Total Units Produced: **{overall_oee_results['total_units_produced']}**
             *   Total Good Units Produced: **{overall_oee_results['total_good_units_produced']}**
             """)

        st.markdown("---")

        if st.checkbox(f"Show Aggregated Data Table ({aggregation_level})"):
            st.subheader(f"Aggregated Data ({aggregation_level})")
            # Display relevant columns from the aggregated data
            group_col_name = "Date" if aggregation_level == "Daily" else ("Year" if aggregation_level == "Yearly" else ("Year_Week" if aggregation_level == "Weekly" else "Year_Month"))

            display_cols = [group_col_name, "OEE (%)", "Availability (%)", "Performance (%)", "Quality (%)",
                            "total_planned_time_minutes", "total_running_time_minutes", "total_downtime_minutes",
                            "total_units_produced", "total_good_units_produced",
                            "total_planned_time_hrs", "total_running_time_hrs", "total_downtime_hrs", # Display hours too
                            "avg_availability_raw", "avg_performance_raw", "avg_quality_raw", "avg_oee_raw" # Avg of raw OEE values
                           ]
            # Filter to only show columns that exist in the aggregated DataFrame
            display_cols = [col for col in display_cols if col in df_aggregated_ts.columns]

            st.dataframe(df_aggregated_ts[display_cols].round(2))


with tab2:
    st.header("Conversational AI")
    st.write(
        f"""
    Ask me specific questions about OEE metrics using filters like Machine ID, Month, or Year.
    Example: "What is the OEE for machine 1 in January 2024?" or "Tell me the Availability for machine 3 in 2025".
    The OEE values presented here are recalculated from the total planned time, running time, and good/total parts for the selected period.
    Note: Performance calculation requires an assumed ideal cycle time (currently {IDEAL_CYCLE_TIME_SECONDS_APP} seconds per part).
    """
    )

    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Handle user input
    if prompt := st.chat_input(
        "Ask about OEE (e.g., 'What's the OEE for machine 3 in February 2025?')", key="chat_input"
    ):
        # Add user message to state and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        chat_history_obj.add_user_message(prompt) # Add to LangChain history

        with st.chat_message("user"):
            st.markdown(prompt)

        # Get and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response = agent_with_chat_history.invoke({"input": prompt})
                assistant_response = response["output"]

            st.markdown(assistant_response)
            st.session_state.messages.append(
                {"role": "assistant", "content": assistant_response}
            )
            chat_history_obj.add_ai_message(assistant_response)