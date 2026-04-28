Read the RDD022_Potholes_SAM3/audit_results_Caio.csv and RDD022_Potholes_SAM3/audit_results_Fernando.csv

Move to RDD022_Potholes_SAM3/RDD022_auditado the images follwing the pattern:

if the image in the .csv is:

nothing: Do not copy the image
segmentation_original: Copy the image with the name and it segmentation in the folder RDD022_Potholes_SAM3/sam_masks
segmentation_cleaned: Copy the image with the name and it segmentation in the folder RDD022_Potholes_SAM3/sam_masks_cleaned
detection: Copy the image with the name and it detection in the folder bboxes


Also add a readme file in the RDD022_Potholes_SAM3/RDD022_auditado folder explaining the structure of the folder and the meaning of the different subfolders.
