IMAGE="US_Olympic__Paralympic_Museum_Quote_522_-_CNA_17_2.jpg"
IMAGE_2="46572_Summary_of_Plan_Offerings_for_RPM_Healthcare_Aetna_0_0.jpg"
IMAGE_3="Monthly_loss_run_522_-_93_KB_5_0.jpg"
IMAGE_4="Manhattan_Christian_CollegePLProposal_03102024_193713__002__3_3.jpg"
IMAGE_5="Loss_Summary_for_BRUSS_NORTH_AMERICA_INC_Valued_as_of_January_17_2022_0_1.jpg"
IMAGE_6="Loss_Summary_for_BRUSS_NORTH_AMERICA_INC_Valued_as_of_January_17_2022_0_0.jpg"
IMAGE_7="FLU_D-QIV-017__201959__Protocol__10-Oct-2014__53_0.jpg"
IMAGE_8="DTPA__BOOSTRIX_-049_BST_048__201334__Protocol_Amendment_1__12-Dec-2016__44_0.jpg"
IMAGE_9="Centurion_ADP_Beneifts_1118_5_0.jpg"
IMAGE_10="Centurion_ADP_Beneifts_1118_1_0.jpg"

SPLIT="val_test"

python demo.py \
    --config-file configs/diffdet.tables.res50.gpu.yaml \
    --input "/home/exouser/DiffusionDet/table-data/recognition30/$SPLIT/images/$IMAGE_10" \
    --output results/ --opts MODEL.WEIGHTS output_tables/model_finetuned.pth MODEL.DEVICE cpu

