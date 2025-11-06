# download aot-ckpt 
gdown --id '1QoChMkTVxdYZ_eBlZhK2acq9KMQZccPJ' --output ./ckpt/R50_DeAOTL_PRE_YTB_DAV.pth

# download sam-ckpt
wget -P ../ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
wget -P ../ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
wget -P ../ckpt https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# download grounding-dino ckpt
wget -P ../ckpt https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
wget -P ../ckpt https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha2/groundingdino_swinb_cogcoor.pth
