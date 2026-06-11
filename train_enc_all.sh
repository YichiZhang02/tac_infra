# vit-B
bash train_enc.sh "rm_nist_260520_usb rm_nist_260323_plug2_raw" scratch vit_b 8 128 100
bash train_enc.sh "rm_nist_260520_usb rm_nist_260323_plug2_raw" clip vit_b 8 128 100

bash train_enc.sh "rm_nist_260520_usb rm_nist_260323_plug2_raw" scratch vit_l 8 128 100
bash train_enc.sh "rm_nist_260520_usb rm_nist_260323_plug2_raw" clip vit_l 8 128 100
bash train_enc.sh "rm_nist_260520_usb rm_nist_260323_plug2_raw" anytouch vit_l 8 128 100