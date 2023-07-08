zip -r frc2023dataset.zip frc2023dataset
scp frc2023dataset.zip team4169@api.seanmabli.com:~/datasets/frc2023dataset.zip
rm frc2023dataset.zip
ssh team4169@api.seanmabli.com "unzip -o datasets/frc2023dataset.zip -d datasets/frc2023dataset; git add -A; git commit -m 'Update dataset'"