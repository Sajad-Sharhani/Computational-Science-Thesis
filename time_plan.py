import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import matplotlib.patches as patches

# Setting a standard, widely available font
plt.rcParams['font.family'] = 'Arial'

# Define the tasks, start and end dates, and colors
tasks_adjusted = {
    "Task": ["Reading Relevant Literature", "Report Writing", "Design and Coding", 
             "Comparison with Existing Approaches", "Presentation Preparation"],
    "Start": [datetime(2023, 1, 1), datetime(2023, 2, 1), datetime(2023, 3, 1), 
              datetime(2023, 4, 1), datetime(2023, 6, 1)],
    "Finish": [datetime(2023, 2, 28), datetime(2023, 6, 30), datetime(2023, 5, 31), 
               datetime(2023, 5, 31), datetime(2023, 6, 30)],
    "Color": ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62228"]
}

# Create a DataFrame from the data
df_adjusted = pd.DataFrame(tasks_adjusted)

# Calculate padding for the x-axis (e.g., 5 days before and after)
padding_days = 5
start_padding = df_adjusted["Start"].min() - timedelta(days=padding_days)
end_padding = df_adjusted["Finish"].max() + timedelta(days=padding_days)

# Plotting the adjusted Gantt chart
fig, ax = plt.subplots(figsize=(12, 8))

# Setting the x-axis range with added padding
ax.set_xlim(start_padding, end_padding)

# Create a rounded rectangle for each task without border
for i, task in enumerate(df_adjusted["Task"]):
    start_date = df_adjusted.loc[i, "Start"]
    duration = df_adjusted.loc[i, "Finish"] - start_date
    rect = patches.FancyBboxPatch((mdates.date2num(start_date), i - 0.3), duration.days, 0.6,
                                  boxstyle="round,pad=0.1,rounding_size=0.8",
                                  linewidth=0,  # No border
                                  facecolor=df_adjusted.loc[i, "Color"])
    ax.add_patch(rect)

# Adjust the y-axis to fit the task names
ax.set_ylim(-0.5, len(df_adjusted["Task"]) - 0.5)

# Format and label the x-axis to show dates
ax.xaxis_date()
ax.xaxis.set_major_locator(mdates.MonthLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d, %Y"))
plt.xticks(rotation=45)  # Rotating the x-axis labels

plt.title('Thesis Time Plan', fontsize=14)

ax.set_yticks(range(len(df_adjusted["Task"])))
ax.set_yticklabels(df_adjusted["Task"])

# Display grid lines for both x and y axes
plt.grid(True, which='major', axis='both', linestyle='--', linewidth=0.5, color='grey')

# Set the frame color
for spine in ax.spines.values():
    spine.set_edgecolor('grey')

# Improve overall layout
plt.tight_layout()

# Save the plot with higher resolution for better readability
plt.savefig('E:\\Uppsala\\Computational-Science-Thesis\\thesis_time_plan_bold.png', dpi=300)

# Show the plot
plt.show()
