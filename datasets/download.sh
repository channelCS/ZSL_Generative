echo "==> Downoading datasets"
# CUB dataset
echo "==> Downoading CUB dataset zip file"
gdown --id 1pTdH0TSGeuwJY0gf_kl64TjQVeN0fnZd
echo "==> Unzipping CUB dataset zip file"
unzip CUB-20210425T113813Z-001.zip
echo "==> Removing CUB dataset zip file"
rm CUB-20210425T113813Z-001.zip

# AWA dataset
echo "==> Downoading AWA dataset zip file"
gdown --id 1FEJWB5goZEQv4pTAMrJTRzJ-RPpBirzk
echo "==> Unzipping AWA dataset zip file"
unzip AWA2-20210614T103407Z-001.zip
echo "==> Removing AWA dataset zip file"
rm AWA2-20210614T103407Z-001.zip
mv AWA2/ AWA/

# FLO dataset
echo "==> Downoading FLO dataset zip file"
gdown --id 1tJ5pFJaylnHGtajkb3FPwFtVjAx7xs3u
echo "==> Unzipping FLO dataset zip file"
unzip FLO-20210614T101759Z-001.zip
echo "==> Removing FLO dataset zip file"
rm FLO-20210614T101759Z-001.zip

# SUN dataset
echo "==> Downoading SUN dataset zip file"
gdown --id 1IYAGZCq5ghhnmo_2wgvmLW8IEcUNk34X
echo "==> Unzipping SUN dataset zip file"
unzip SUN-20210614T150751Z-001.zip
echo "==> Removing SUN dataset zip file"
rm SUN-20210614T150751Z-001.zip

echo "==> Downoading datasets completed"
