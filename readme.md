老师要求：
Topic 6: Taxi Trajectory Analysis
Taxis are major transportation tools. Analyzing their GPS trajectories provides insights into crowd movement, traffic, regional prosperity, and route recommendations.
	Data Configuration: Download the dataset 01.zip to 014.zip from T-Drive dataset (from Microsoft Research). It contains over 10,000 .txt files recording GPS data of 10,000 Beijing taxis over a week (ID, Time, Longitude, Latitude).
	Functional Requirements:
	F1. Visualization: Display the trajectory of all or specific taxis. You can draw a map in your program or use Google Maps / Baidu Maps APIs.
	F2. Map Zoom: Zoom in/out and adjust trajectory rendering accordingly.
	F3. Regional Search: Count the number of taxis within a user-specified rectangular area (defined by top-left and bottom-right coordinates) during a specific time period.
	F4. Traffic Density Analysis: Given parameter r, divide the map into r\times r grids. Analyze traffic density changes across all grids over different time periods.
	F5. Regional Correlation 1: Specify two rectangular regions, calculate traffic volume traveling between them over different time periods.
	F6. Regional Correlation 2: Specify one rectangular region, calculate traffic volume traveling between this region and all other regions over time.
	F7. Frequent Path 1: Define path frequency as the total number of cars passing through it. Given k and distance x, find the Top k most frequent paths globally that are longer than x.
	F8. Frequent Path 2: Given regions A and B, analyze the Top k most frequent paths from A to B.
	F9. Travel Time Analysis: Given regions A and B, analyze the path with the shortest travel time from A to B across different time periods, and output the estimated time.