# Roadmap

1. Complete E2E EC2 <--> IVS Connection
    - Test delay on EC2 - if any delay, find solutions.
        - Ideally 4K video can be processed real time, otherwise investigate alternatives (1080p, processing frame rate)
    - Refactor code to read from IVS instead of playing video file
        - maybe add a flag for "testing" vs. "live"
    - Test E2E flow recording the screen live
    - Determine backup, video saving from IVS etc

2. Validate Yolo/OCR Performance
    - Build out race list CSV ingestion to names, bib numbers and racer metadata
    - Run E2E on test and record finish list, compare with final results
    - Analyze the results - Does the model perform well on the dataset
    - Does the model perform well outside of training set? Test out of distribution data to determine generalizability
        - if no, need to find different bib datasets
        - ensure fine tuning didn't destroy generalizability

3. Test connectivity
    - Test streaming and processing with no wifi - do we need a hotspot

4. Game day operations
    - Optimize the leaderboard - top x results by y categories to display on a screen somewhere (validate with stakeholders)
    - Answer operational questions
        - How will the camera/computer be set up at the race?
        - Where will the screen be?
    - Stress test system - what can go wrong, mitigations, back-ups and contingencies

# Overall TODOs
## AWS IVS

&#x2611; **Setup IVS** - Establish live connection between mobile device and IVS.

- Established connection between mobile device and IVS using Prism Live Studio.

&#x2610; **Read IVS from EC2** - Build out connection between the IVS and the EC2 instance, validating the E2E pipeline.

&#x2610; **Finalize Streaming** - Ensure streaming software on mobile device is adequate for capturing ~1hour long stream

## Validate Compute

&#x2610; **Determine Minimally Viable EC2** - Determine whch EC2 instance is sufficient for live streaming in 4K/1080p. Do this by simulating stream from video file.

## Modeling

&#x2610; **More Training** - Pull in additional open source bib data sources to bolster training data for more robust model.

## Miscellaneous

&#x2610; **Architecture Diagram** - Build out architecture diagram for readme

&#x2610; **Validate Upload Requirements** - Determine necessary upload requirements for live edge streaming of video and determine what hardware is required. Is hotspot adequate?

## Code

&#x2610; **Add Database with APIs** - Add pipeline component to save leaderboard to a database, determining the optimal option (DynamoDB?).

&#x2610; **Add Frontend** - Build out front end components to allow users to update bib information, fill in missing bibs, add racers that the software missed, etc.

&#x2610; **Add Utility Backend APIs** - Build out backend APIs for various backend functions e.g., to read/update data in database

&#x2610; **Connect to Master CSV** - Build out a method for linking the data directly to the master csv to pull information about users. 

&#x2610; **Develop a Leaderboard** - Build out a leaderboard with results segrated by overall, age bracked, gender.

## Cost
&#x2610; **Cost Tracking** - Compile a running list of components and track the cost of each to keep operational costs low and elliminate inefficiencies.


## Testing
&#x2610; **Field Test** - Test connection, leaderboard, backend and frontend code for an hour at location.
