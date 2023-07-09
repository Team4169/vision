echo GitHub Username:
read varname
zip -r frc2023dataset.zip frc2023dataset
scp frc2023dataset.zip team4169@api.seanmabli.com:~/datasets/frc2023dataset.zip
rm frc2023dataset.zip
ssh team4169@api.seanmabli.com "cd datasets; rm -fr frc2023dataset/; unzip -o frc2023dataset.zip -d .; git add -A; git commit -m 'update dataset - $varname'"