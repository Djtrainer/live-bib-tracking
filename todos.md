# Kinesis Data Streams
- **Setup KDS** - Establish live connection between mobile device and KDS.
- **Read KDS from EC2** - Build out connection between the KDS and the EC2 instance, validating the E2E pipeline.

# Validate Compute
- **Determine Minimally Viable EC2** - Determine whch EC2 instance is sufficient for live streaming in 4K/1080p. Do this by simulating stream from video file.

# Modeling
- **More Training** - Pull in additional open source bib data sources to bolster training data for more robust model.

# Miscellaneous
- **Validate Upload Requirements** - Determine necessary upload requirements for live edge streaming of video and determine what hardware is required. Is hotspot adequate?

# Code
- **Add Database with APIs** - Add pipeline component to save leaderboard to a database, determining the optimal option (DynamoDB?).
- **Add Frontend** - Build out front end components to allow users to update bib information, fill in missing bibs, add racers that the software missed, etc.
- **Add Utility Backend APIs** - Build out backend APIs for various backend functions e.g., to read/update data in database

# Cost
- **Cost Tracking** - Compile a running list of components and track the cost of each to keep operational costs low and elliminate inefficiencies.