================================
ADAM Quality Control (QC) Viewer
================================

Documentation/Guide:

https://goeden-gab.notion.site/Data-Quality-Control-QC-Review-Tool-21c7c1ad9aee800b85bcdb2e3814e3f5 


============================== 
What is this?

This is a simple, standalone tool for reviewing documents. It runs entirely in your web browser.

You use it to open a folder of JSON files, check them one by one, and then create a report that summarizes which documents you 'Approved' and which you 'Rejected'.

How to Use It

Step 1: Save the qc-viewer-app.html file to your computer (e.g., your Desktop).

Step 2: Double-click the qc-viewer-app.html file to open it in your web browser (like Chrome or Firefox).

Step 3: Click the "Select Folder to Review" button and choose the folder that contains your JSON documents.

Step 4: The application will load. Enter your name in the "Data QC Operator Name Here" box at the top.

Step 5: Click on a document from the list on the left to view it on the right.

Step 6: Use the "Approve" or "Reject" buttons for each document. If you reject a document, you must provide a reason.

Step 7: Once you have reviewed all the documents, click the "Generate Report" button to create your summary. You can then copy this report or download it as a file.

Required JSON File Format

For the tool to read your files, each .json file must contain at least a "document_id" and a "content_blocks" section. Files that do not have these will be skipped.

Example:
{
"document_id": "some-unique-name",
"source_file": "source.pdf",
"metadata": { ... },
"content_blocks": [ ... ]
}