import tqdm as tqdm
from urllib3 import PoolManager
from pathlib import Path
import logging
from tqdm import tqdm
urls = ['https://zenodo.org/record/4746605/files/2018_BART_4_322000_4882000_image_crop.laz?download=1',
        'https://zenodo.org/record/4746605/files/2018_BART_4_322000_4882000_image_crop.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_BART_4_322000_4882000_image_crop_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_BART_4_322000_4882000_image_crop_hyperspectral.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_HARV_5_733000_4698000_image_crop.laz?download=1',
        'https://zenodo.org/record/4746605/files/2018_HARV_5_733000_4698000_image_crop.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_HARV_5_733000_4698000_image_crop_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_HARV_5_733000_4698000_image_crop_hyperspectral.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_JERC_4_742000_3451000_image_crop.laz?download=1',
        'https://zenodo.org/record/4746605/files/2018_JERC_4_742000_3451000_image_crop.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_JERC_4_742000_3451000_image_crop_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_JERC_4_742000_3451000_image_crop_hyperspectral.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_MLBS_3_541000_4140000_image_crop.laz?download=1',
        'https://zenodo.org/record/4746605/files/2018_MLBS_3_541000_4140000_image_crop.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_MLBS_3_541000_4140000_image_crop2.laz?download=1',
        'https://zenodo.org/record/4746605/files/2018_MLBS_3_541000_4140000_image_crop2.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_MLBS_3_541000_4140000_image_crop2_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_MLBS_3_541000_4140000_image_crop2_hyperspectral.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_MLBS_3_541000_4140000_image_crop_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_MLBS_3_541000_4140000_image_crop_hyperspectral.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_NIWO_2_450000_4426000_image_crop.laz?download=1',
        'https://zenodo.org/record/4746605/files/2018_NIWO_2_450000_4426000_image_crop.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_NIWO_2_450000_4426000_image_crop_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_NIWO_2_450000_4426000_image_crop_hyperspectral.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_OSBS_4_405000_3286000_image.laz?download=1',
        'https://zenodo.org/record/4746605/files/2018_OSBS_4_405000_3286000_image.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_OSBS_4_405000_3286000_image_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_SJER_3_258000_4106000_image.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_SJER_3_258000_4106000_image_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_SJER_3_259000_4110000_image.laz?download=1',
        'https://zenodo.org/record/4746605/files/2018_SJER_3_259000_4110000_image.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_SJER_3_259000_4110000_image_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2018_TEAK_3_315000_4094000_image_crop.laz?download=1',
        'https://zenodo.org/record/4746605/files/2018_TEAK_3_315000_4094000_image_crop_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_DELA_5_423000_3601000_image_crop.laz?download=1',
        'https://zenodo.org/record/4746605/files/2019_DELA_5_423000_3601000_image_crop.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_DELA_5_423000_3601000_image_crop_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_DELA_5_423000_3601000_image_crop_hyperspectral.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_LENO_5_383000_3523000_image_crop.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_LENO_5_383000_3523000_image_crop_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_LENO_5_383000_3523000_image_crop_hyperspectral.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_OSBS_5_405000_3287000_image2.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_OSBS_5_405000_3287000_image2_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_OSBS_5_405000_3287000_image2_hyperspectral.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_OSBS_5_405000_3287000_image_crop.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_OSBS_5_405000_3287000_image_crop2.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_OSBS_5_405000_3287000_image_crop2_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_OSBS_5_405000_3287000_image_crop2_hyperspectral.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_OSBS_5_405000_3287000_image_crop_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_OSBS_5_405000_3287000_image_crop_hyperspectral.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_SJER_4_251000_4103000_image.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_SJER_4_251000_4103000_image_CHM.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_TOOL_3_403000_7617000_image.tif?download=1',
        'https://zenodo.org/record/4746605/files/2019_TOOL_3_403000_7617000_image_CHM.tif?download=1']


def download_data(url_list, save_dir):
    pm = PoolManager()
    for url in tqdm(url_list):
        filename = url.split('/')[-1].split('?')[0]
        filename = Path(f'{save_dir}/{filename}')
        logging.log(logging.INFO, f'Downloading and saving to {filename}...')
        if filename.exists():
            logging.log(logging.INFO, f'Operarions were skipped for {filename}!!!')
            continue
        with pm.request('GET', url, preload_content=False) as resp, open(filename, 'wb') as out_file:
            out_file.write(resp.read())


if __name__ == '__main__':
    download_data(urls, 'raw')


def generate_urls():
    filenames = """2018_BART_4_322000_4882000_image_crop.laz

        md5:dda4b19cdc50a8bce71834a0ea8324d0 	328.7 kB	
        2018_BART_4_322000_4882000_image_crop.tif

        md5:06bccf59dc274b97494466cc62c1a5a5 	3.8 MB	 
        2018_BART_4_322000_4882000_image_crop_CHM.tif

        md5:aafc0880e079e8f36c7e44c8cdd0e74a 	42.2 kB	 
        2018_BART_4_322000_4882000_image_crop_hyperspectral.tif

        md5:7691073bd6eb9a10bc456685fb87d371 	7.6 MB	 
        2018_HARV_5_733000_4698000_image_crop.laz

        md5:1281d93764ddbd52cbd9348f7754459e 	638.5 kB	
        2018_HARV_5_733000_4698000_image_crop.tif

        md5:a8dbe47fbdb5d94b281e3dbc2ec76e27 	8.0 MB	 
        2018_HARV_5_733000_4698000_image_crop_CHM.tif

        md5:b267df7e42df6bb4d021e3e92ca600da 	90.9 kB	 
        2018_HARV_5_733000_4698000_image_crop_hyperspectral.tif

        md5:f543223a692a694a184d8053bb6a9380 	16.4 MB	 
        2018_JERC_4_742000_3451000_image_crop.laz

        md5:3dc1ea61592acbf40c8d619aeff0f4b1 	300.4 kB	
        2018_JERC_4_742000_3451000_image_crop.tif

        md5:a34cbf2c1e50e3dbbd987919dad93d1b 	5.8 MB	 
        2018_JERC_4_742000_3451000_image_crop_CHM.tif

        md5:ec55fba1338cc78a6c131ac9591227f5 	65.9 kB	 
        2018_JERC_4_742000_3451000_image_crop_hyperspectral.tif

        md5:2b3967d0bdfb31b851efa3385ed89315 	11.9 MB	 
        2018_MLBS_3_541000_4140000_image_crop.laz

        md5:014e5ffc3a3af46d89707e661f8475bd 	3.9 MB	
        2018_MLBS_3_541000_4140000_image_crop.tif

        md5:01a7cf23b368ff9e006fda8fe9ca4c8c 	10.0 MB	 
        2018_MLBS_3_541000_4140000_image_crop2.laz

        md5:8063710cff297a28fcd63a71f6feb5f2 	6.5 MB	
        2018_MLBS_3_541000_4140000_image_crop2.tif

        md5:48b70255ee9c4cee436620829f44df12 	13.1 MB	 
        2018_MLBS_3_541000_4140000_image_crop2_CHM.tif

        md5:02f9158384eef95e2efca97357ed91b2 	144.3 kB	 
        2018_MLBS_3_541000_4140000_image_crop2_hyperspectral.tif

        md5:9be678a66abd12a7b88ee78cded8d7c1 	26.2 MB	 
        2018_MLBS_3_541000_4140000_image_crop_CHM.tif

        md5:9dcd2b67f7e4557d091d0891e9c92c57 	108.0 kB	 
        2018_MLBS_3_541000_4140000_image_crop_hyperspectral.tif

        md5:7e21bf6853df70c065accf9aa481eb58 	19.6 MB	 
        2018_NIWO_2_450000_4426000_image_crop.laz

        md5:adb2a40470af9de4acea1bed8ce1cac2 	15.3 MB	
        2018_NIWO_2_450000_4426000_image_crop.tif

        md5:c8f700eca920c6f0b93d16e6e26cc5a7 	36.5 MB	 
        2018_NIWO_2_450000_4426000_image_crop_CHM.tif

        md5:4830d55fd48a1875675cc976cfeead21 	418.0 kB	 
        2018_NIWO_2_450000_4426000_image_crop_hyperspectral.tif

        md5:1b1326116dec6d6d20f44b2a8d803c99 	76.5 MB	 
        2018_OSBS_4_405000_3286000_image.laz

        md5:b6ced7d9427c8a5945bf11c0a9233e90 	8.7 MB	
        2018_OSBS_4_405000_3286000_image.tif

        md5:f3759a54251ca0cf8c05fa1f47c27179 	73.3 MB	 
        2018_OSBS_4_405000_3286000_image_CHM.tif

        md5:c9bf4cc5e9070c5fab0a92b6b681151f 	4.0 MB	 
        2018_SJER_3_258000_4106000_image.tif

        md5:d70ecbee40abe043946e8e492c514a63 	54.0 MB	 
        2018_SJER_3_258000_4106000_image_CHM.tif

        md5:78d5bc08ffaa083fc95240266d17c206 	4.0 MB	 
        2018_SJER_3_259000_4110000_image.laz

        md5:ad5549a021ced41c59e96a587fb4d37d 	225.1 MB	
        2018_SJER_3_259000_4110000_image.tif

        md5:79b3804e212761275f8612b1fc3f0a8f 	57.9 MB	 
        2018_SJER_3_259000_4110000_image_CHM.tif

        md5:d563357a393e69a2e46a4915f7a7059b 	4.0 MB	 
        2018_TEAK_3_315000_4094000_image_crop.laz

        md5:cf8101b0ca90b687c3f270208117a191 	24.2 MB	
        2018_TEAK_3_315000_4094000_image_crop_CHM.tif

        md5:6976263ff0a84531b333517e244564eb 	1.4 MB	 
        2019_DELA_5_423000_3601000_image_crop.laz

        md5:7533fbba990e973e7375e2c0013bac0f 	521.7 kB	
        2019_DELA_5_423000_3601000_image_crop.tif

        md5:4548d77c9d040321faa7b86478354ea7 	6.0 MB	 
        2019_DELA_5_423000_3601000_image_crop_CHM.tif

        md5:005a1f7de93c5fc7893590c0a38c03c1 	72.3 kB	 
        2019_DELA_5_423000_3601000_image_crop_hyperspectral.tif

        md5:031cd78f0249438e9e577506a2c33623 	13.1 MB	 
        2019_LENO_5_383000_3523000_image_crop.tif

        md5:9316b2bafa2347d17ce0e98c9aae57fb 	9.6 MB	 
        2019_LENO_5_383000_3523000_image_crop_CHM.tif

        md5:32dffacf26562b8b15e83d07e6c712c4 	108.8 kB	 
        2019_LENO_5_383000_3523000_image_crop_hyperspectral.tif

        md5:85fa45179428c2793cf35ac765efedf7 	20.0 MB	 
        2019_OSBS_5_405000_3287000_image2.tif

        md5:fe14cd127ef4d808d9e7d07b590c024d 	21.6 MB	 
        2019_OSBS_5_405000_3287000_image2_CHM.tif

        md5:9b65c9b11beaf85f14c685bed18c27de 	241.3 kB	 
        2019_OSBS_5_405000_3287000_image2_hyperspectral.tif

        md5:17ebbbc77481b23087aafc3a7e6a5763 	44.2 MB	 
        2019_OSBS_5_405000_3287000_image_crop.tif

        md5:23966808231e1a5af3b5984631cacad8 	10.9 MB	 
        2019_OSBS_5_405000_3287000_image_crop2.tif

        md5:fe14cd127ef4d808d9e7d07b590c024d 	21.6 MB	 
        2019_OSBS_5_405000_3287000_image_crop2_CHM.tif

        md5:9b65c9b11beaf85f14c685bed18c27de 	241.3 kB	 
        2019_OSBS_5_405000_3287000_image_crop2_hyperspectral.tif

        md5:17ebbbc77481b23087aafc3a7e6a5763 	44.2 MB	 
        2019_OSBS_5_405000_3287000_image_crop_CHM.tif

        md5:79d51cc6db438c65b5b7a7414978452d 	123.2 kB	 
        2019_OSBS_5_405000_3287000_image_crop_hyperspectral.tif

        md5:17e7b7d322cefc36272f55aa1d8eea91 	22.4 MB	 
        2019_SJER_4_251000_4103000_image.tif

        md5:eea10d7d68da7d3b345ce99678d198d9 	25.5 MB	 
        2019_SJER_4_251000_4103000_image_CHM.tif

        md5:4cf3d4ab986a80ee35570a2c695d0845 	4.0 MB	 
        2019_TOOL_3_403000_7617000_image.tif

        md5:1ebf5b09cf5deafa4e2c16b3721f176a 	77.1 MB	 
        2019_TOOL_3_403000_7617000_image_CHM.tif

        md5:91dc29a3795306c8ca1863049bcad033 	"""

    return list(map(lambda x: f'https://zenodo.org/record/4746605/files/{x}?download=1',
                    filter(lambda x: len(x) and not x.startswith('md5:'), filenames.splitlines())))
