"""Module of sample legends for some commonly used geospatial datasets.
"""

import os


# Land Cover datasets in Earth Engine https://developers.google.com/earth-engine/datasets/tags/landcover
builtin_legends = {
    # National Land Cover Database 2016 (NLCD2016) Legend https://www.mrlc.gov/data/legends/national-land-cover-database-2016-nlcd2016-legend
    "NLCD": {
        "11 Open Water": "466b9f",
        "12 Perennial Ice/Snow": "d1def8",
        "21 Developed, Open Space": "dec5c5",
        "22 Developed, Low Intensity": "d99282",
        "23 Developed, Medium Intensity": "eb0000",
        "24 Developed, High Intensity": "ab0000",
        "31 Barren Land (Rock/Sand/Clay)": "b3ac9f",
        "41 Deciduous Forest": "68ab5f",
        "42 Evergreen Forest": "1c5f2c",
        "43 Mixed Forest": "b5c58f",
        "51 Dwarf Scrub": "af963c",
        "52 Shrub/Scrub": "ccb879",
        "71 Grassland/Herbaceous": "dfdfc2",
        "72 Sedge/Herbaceous": "d1d182",
        "73 Lichens": "a3cc51",
        "74 Moss": "82ba9e",
        "81 Pasture/Hay": "dcd939",
        "82 Cultivated Crops": "ab6c28",
        "90 Woody Wetlands": "b8d9eb",
        "95 Emergent Herbaceous Wetlands": "6c9fb8",
    },
    # https://developers.google.com/earth-engine/datasets/catalog/ESA_WorldCover_v100
    "ESA_WorldCover": {
        "10 Trees": "006400",
        "20 Shrubland": "ffbb22",
        "30 Grassland": "ffff4c",
        "40 Cropland": "f096ff",
        "50 Built-up": "fa0000",
        "60 Barren / sparse vegetation": "b4b4b4",
        "70 Snow and ice": "f0f0f0",
        "80 Open water": "0064c8",
        "90 Herbaceous wetland": "0096a0",
        "95 Mangroves": "00cf75",
        "100 Moss and lichen": "fae6a0",
    },
    # https://samapriya.github.io/awesome-gee-community-datasets/projects/esrilc2020/
    "ESRI_LandCover": {
        "1 Water": "1A5BAB",
        "2 Trees": "358221",
        "3 Grass": "A7D282",
        "4 Flooded Vegetation": "87D19E",
        "5 Crops": "FFDB5C",
        "6 Shrub & Scrub": "EECFA8",
        "7 Built Area": "ED022A",
        "8 Bare Ground": "EDE9E4",
        "9 Snow/Ice": "F2FAFF",
        "10 Clouds": "C8C8C8",
    },
    # https://samapriya.github.io/awesome-gee-community-datasets/projects/S2TSLULC/
    "ESRI_LandCover_TS": {
        "1 Water": "1A5BAB",
        "2 Trees": "358221",
        "4 Flooded Vegetation": "87D19E",
        "5 Crops": "FFDB5C",
        "7 Built Area": "ED022A",
        "8 Bare Ground": "EDE9E4",
        "9 Snow/Ice": "F2FAFF",
        "10 Clouds": "C8C8C8",
        "11 Rangeland": "C6AD8D",
    },
    "Dynamic_World": {
        "0 Water": "419BDF",
        "1 Trees": "397D49",
        "2 Grass": "88B053",
        "3 Flooded vegetation": "7A87C6",
        "4 Crops": "E49635",
        "5 Shrub & Scrub": "DFC35A",
        "6 Built-up area": "C4281B",
        "7 Bare ground": "A59B8F",
        "8 Snow & Ice": "B39FE1",
    },
    # National Wetlands Inventory Legend: https://www.fws.gov/wetlands/data/Mapper-Wetlands-Legend.html
    "NWI": {
        "Freshwater- Forested and Shrub wetland": (0, 136, 55),
        "Freshwater Emergent wetland": (127, 195, 28),
        "Freshwater pond": (104, 140, 192),
        "Estuarine and Marine wetland": (102, 194, 165),
        "Riverine": (1, 144, 191),
        "Lakes": (19, 0, 124),
        "Estuarine and Marine Deepwater": (0, 124, 136),
        "Other Freshwater wetland": (178, 134, 86),
    },
    # MCD12Q1.051 Land Cover Type Yearly Global 500m https://developers.google.com/earth-engine/datasets/catalog/MODIS_051_MCD12Q1
    "MODIS/051/MCD12Q1": {
        "0 Water": "1c0dff",
        "1 Evergreen needleleaf forest": "05450a",
        "2 Evergreen broadleaf forest": "086a10",
        "3 Deciduous needleleaf forest": "54a708",
        "4 Deciduous broadleaf forest": "78d203",
        "5 Mixed forest": "009900",
        "6 Closed shrublands": "c6b044",
        "7 Open shrublands": "dcd159",
        "8 Woody savannas": "dade48",
        "9 Savannas": "fbff13",
        "10 Grasslands": "b6ff05",
        "11 Permanent wetlands": "27ff87",
        "12 Croplands": "c24f44",
        "13 Urban and built-up": "a5a5a5",
        "14 Cropland/natural vegetation mosaic": "ff6d4c",
        "15 Snow and ice": "69fff8",
        "16 Barren or sparsely vegetated": "f9ffa4",
        "254 Unclassified": "ffffff",
    },
    # MCD12Q1.006 Land Cover Type Yearly Global 500m https://developers.google.com/earth-engine/datasets/catalog/MODIS_006_MCD12Q1
    "MODIS/006/MCD12Q1": {
        "1 Evergreen needleleaf forest": "05450a",
        "2 Evergreen broadleaf forest": "086a10",
        "3 Deciduous needleleaf forest": "54a708",
        "4 Deciduous broadleaf forest": "78d203",
        "5 Mixed forest": "009900",
        "6 Closed shrublands": "c6b044",
        "7 Open shrublands": "dcd159",
        "8 Woody savannas": "dade48",
        "9 Savannas": "fbff13",
        "10 Grasslands": "b6ff05",
        "11 Permanent wetlands": "27ff87",
        "12 Croplands": "c24f44",
        "13 Urban and built-up": "a5a5a5",
        "14 Cropland/natural vegetation mosaic": "ff6d4c",
        "15 Snow and ice": "69fff8",
        "16 Barren or sparsely vegetated": "f9ffa4",
        "17 Water bodies": "1c0dff",
    },
    # GlobCover: Global Land Cover Map https://developers.google.com/earth-engine/datasets/catalog/ESA_GLOBCOVER_L4_200901_200912_V2_3
    "GLOBCOVER": {
        "11 Post-flooding or irrigated croplands": "aaefef",
        "14 Rainfed croplands": "ffff63",
        "20 Mosaic cropland (50-70%) / vegetation (grassland, shrubland, forest) (20-50%)": "dcef63",
        "30 Mosaic vegetation (grassland, shrubland, forest) (50-70%) / cropland (20-50%)": "cdcd64",
        "40 Closed to open (>15%) broadleaved evergreen and/or semi-deciduous forest (>5m)": "006300",
        "50 Closed (>40%) broadleaved deciduous forest (>5m)": "009f00",
        "60 Open (15-40%) broadleaved deciduous forest (>5m)": "aac700",
        "70 Closed (>40%) needleleaved evergreen forest (>5m)": "003b00",
        "90 Open (15-40%) needleleaved deciduous or evergreen forest (>5m)": "286300",
        "100 Closed to open (>15%) mixed broadleaved and needleleaved forest (>5m)": "788300",
        "110 Mosaic forest-shrubland (50-70%) / grassland (20-50%)": "8d9f00",
        "120 Mosaic grassland (50-70%) / forest-shrubland (20-50%)": "bd9500",
        "130 Closed to open (>15%) shrubland (<5m)": "956300",
        "140 Closed to open (>15%) grassland": "ffb431",
        "150 Sparse (>15%) vegetation (woody vegetation, shrubs, grassland)": "ffebae",
        "160 Closed (>40%) broadleaved forest regularly flooded - Fresh water": "00785a",
        "170 Closed (>40%) broadleaved semi-deciduous and/or evergreen forest regularly flooded - saline water": "009578",
        "180 Closed to open (>15%) vegetation (grassland, shrubland, woody vegetation) on regularly flooded or waterlogged soil - fresh, brackish or saline water": "00dc83",
        "190 Artificial surfaces and associated areas (urban areas >50%) GLOBCOVER 2009": "c31300",
        "200 Bare areas": "fff5d6",
        "210 Water bodies": "0046c7",
        "220 Permanent snow and ice": "ffffff",
        "230 Unclassified": "743411",
    },
    # Global PALSAR-2/PALSAR Forest/Non-Forest Map https://developers.google.com/earth-engine/datasets/catalog/JAXA_ALOS_PALSAR_YEARLY_FNF
    "JAXA/PALSAR": {
        "1 Forest": "006400",
        "2 Non-Forest": "FEFF99",
        "3 Water": "0000FF",
    },
    # Oxford MAP: Malaria Atlas Project Fractional International Geosphere-Biosphere Programme Landcover https://developers.google.com/earth-engine/datasets/catalog/Oxford_MAP_IGBP_Fractional_Landcover_5km_Annual
    "Oxford": {
        "0 Water": "032f7e",
        "1 Evergreen_Needleleaf_Fores": "02740b",
        "2 Evergreen_Broadleaf_Forest": "02740b",
        "3 Deciduous_Needleleaf_Forest": "8cf502",
        "4 Deciduous_Broadleaf_Forest": "8cf502",
        "5 Mixed_Forest": "a4da01",
        "6 Closed_Shrublands": "ffbd05",
        "7 Open_Shrublands": "ffbd05",
        "8 Woody_Savannas": "7a5a02",
        "9 Savannas": "f0ff0f",
        "10 Grasslands": "869b36",
        "11 Permanent_Wetlands": "6091b4",
        "12 Croplands": "ff4e4e",
        "13 Urban_and_Built-up": "999999",
        "14 Cropland_Natural_Vegetation_Mosaic": "ff4e4e",
        "15 Snow_and_Ice": "ffffff",
        "16 Barren_Or_Sparsely_Vegetated": "feffc0",
        "17 Unclassified": "020202",
    },
    # Canada AAFC Annual Crop Inventory https://developers.google.com/earth-engine/datasets/catalog/AAFC_ACI
    "AAFC/ACI": {
        "10 Cloud": "000000",
        "20 Water": "3333ff",
        "30 Exposed Land and Barren": "996666",
        "34 Urban and Developed": "cc6699",
        "35 Greenhouses": "e1e1e1",
        "50 Shrubland": "ffff00",
        "80 Wetland": "993399",
        "110 Grassland": "cccc00",
        "120 Agriculture (undifferentiated)": "cc6600",
        "122 Pasture and Forages": "ffcc33",
        "130 Too Wet to be Seeded": "7899f6",
        "131 Fallow": "ff9900",
        "132 Cereals": "660000",
        "133 Barley": "dae31d",
        "134 Other Grains": "d6cc00",
        "135 Millet": "d2db25",
        "136 Oats": "d1d52b",
        "137 Rye": "cace32",
        "138 Spelt": "c3c63a",
        "139 Triticale": "b9bc44",
        "140 Wheat": "a7b34d",
        "141 Switchgrass": "b9c64e",
        "142 Sorghum": "999900",
        "145 Winter Wheat": "92a55b",
        "146 Spring Wheat": "809769",
        "147 Corn": "ffff99",
        "148 Tobacco": "98887c",
        "149 Ginseng": "799b93",
        "150 Oilseeds": "5ea263",
        "151 Borage": "52ae77",
        "152 Camelina": "41bf7a",
        "153 Canola and Rapeseed": "d6ff70",
        "154 Flaxseed": "8c8cff",
        "155 Mustard": "d6cc00",
        "156 Safflower": "ff7f00",
        "157 Sunflower": "315491",
        "158 Soybeans": "cc9933",
        "160 Pulses": "896e43",
        "162 Peas": "8f6c3d",
        "167 Beans": "82654a",
        "174 Lentils": "b85900",
        "175 Vegetables": "b74b15",
        "176 Tomatoes": "ff8a8a",
        "177 Potatoes": "ffcccc",
        "178 Sugarbeets": "6f55ca",
        "179 Other Vegetables": "ffccff",
        "180 Fruits": "dc5424",
        "181 Berries": "d05a30",
        "182 Blueberry": "d20000",
        "183 Cranberry": "cc0000",
        "185 Other Berry": "dc3200",
        "188 Orchards": "ff6666",
        "189 Other Fruits": "c5453b",
        "190 Vineyards": "7442bd",
        "191 Hops": "ffcccc",
        "192 Sod": "b5fb05",
        "193 Herbs": "ccff05",
        "194 Nursery": "07f98c",
        "195 Buckwheat": "00ffcc",
        "196 Canaryseed": "cc33cc",
        "197 Hemp": "8e7672",
        "198 Vetch": "b1954f",
        "199 Other Crops": "749a66",
        "200 Forest (undifferentiated)": "009900",
        "210 Coniferous": "006600",
        "220 Broadleaf": "00cc00",
        "230 Mixedwood": "cc9900",
    },
    # Copernicus CORINE Land Cover https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_CORINE_V20_100m
    "COPERNICUS/CORINE/V20/100m": {
        "111 Artificial surfaces > Urban fabric > Continuous urban fabric": "E6004D",
        "112 Artificial surfaces > Urban fabric > Discontinuous urban fabric": "FF0000",
        "121 Artificial surfaces > Industrial, commercial, and transport units > Industrial or commercial units": "CC4DF2",
        "122 Artificial surfaces > Industrial, commercial, and transport units > Road and rail networks and associated land": "CC0000",
        "123 Artificial surfaces > Industrial, commercial, and transport units > Port areas": "E6CCCC",
        "124 Artificial surfaces > Industrial, commercial, and transport units > Airports": "E6CCE6",
        "131 Artificial surfaces > Mine, dump, and construction sites > Mineral extraction sites": "A600CC",
        "132 Artificial surfaces > Mine, dump, and construction sites > Dump sites": "A64DCC",
        "133 Artificial surfaces > Mine, dump, and construction sites > Construction sites": "FF4DFF",
        "141 Artificial surfaces > Artificial, non-agricultural vegetated areas > Green urban areas": "FFA6FF",
        "142 Artificial surfaces > Artificial, non-agricultural vegetated areas > Sport and leisure facilities": "FFE6FF",
        "211 Agricultural areas > Arable land > Non-irrigated arable land": "FFFFA8",
        "212 Agricultural areas > Arable land > Permanently irrigated land": "FFFF00",
        "213 Agricultural areas > Arable land > Rice fields": "E6E600",
        "221 Agricultural areas > Permanent crops > Vineyards": "E68000",
        "222 Agricultural areas > Permanent crops > Fruit trees and berry plantations": "F2A64D",
        "223 Agricultural areas > Permanent crops > Olive groves": "E6A600",
        "231 Agricultural areas > Pastures > Pastures": "E6E64D",
        "241 Agricultural areas > Heterogeneous agricultural areas > Annual crops associated with permanent crops": "FFE6A6",
        "242 Agricultural areas > Heterogeneous agricultural areas > Complex cultivation patterns": "FFE64D",
        "243 Agricultural areas > Heterogeneous agricultural areas > Land principally occupied by agriculture, with significant areas of natural vegetation": "E6CC4D",
        "244 Agricultural areas > Heterogeneous agricultural areas > Agro-forestry areas": "F2CCA6",
        "311 Forest and semi natural areas > Forests > Broad-leaved forest": "80FF00",
        "312 Forest and semi natural areas > Forests > Coniferous forest": "00A600",
        "313 Forest and semi natural areas > Forests > Mixed forest": "4DFF00",
        "321 Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Natural grasslands": "CCF24D",
        "322 Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Moors and heathland": "A6FF80",
        "323 Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Sclerophyllous vegetation": "A6E64D",
        "324 Forest and semi natural areas > Scrub and/or herbaceous vegetation associations > Transitional woodland-shrub": "A6F200",
        "331 Forest and semi natural areas > Open spaces with little or no vegetation > Beaches, dunes, sands": "E6E6E6",
        "332 Forest and semi natural areas > Open spaces with little or no vegetation > Bare rocks": "CCCCCC",
        "333 Forest and semi natural areas > Open spaces with little or no vegetation > Sparsely vegetated areas": "CCFFCC",
        "334 Forest and semi natural areas > Open spaces with little or no vegetation > Burnt areas": "000000",
        "335 Forest and semi natural areas > Open spaces with little or no vegetation > Glaciers and perpetual snow": "A6E6CC",
        "411 Wetlands > Inland wetlands > Inland marshes": "A6A6FF",
        "412 Wetlands > Inland wetlands > Peat bogs": "4D4DFF",
        "421 Wetlands > Maritime wetlands > Salt marshes": "CCCCFF",
        "422 Wetlands > Maritime wetlands > Salines": "E6E6FF",
        "423 Wetlands > Maritime wetlands > Intertidal flats": "A6A6E6",
        "511 Water bodies > Inland waters > Water courses": "00CCF2",
        "512 Water bodies > Inland waters > Water bodies": "80F2E6",
        "521 Water bodies > Marine waters > Coastal lagoons": "00FFA6",
        "522 Water bodies > Marine waters > Estuaries": "A6FFE6",
        "523 Water bodies > Marine waters > Sea and ocean": "E6F2FF",
    },
    # Copernicus Global Land Cover Layers: CGLS-LC100 collection 2 https://developers.google.com/earth-engine/datasets/catalog/COPERNICUS_Landcover_100m_Proba-V_Global
    "COPERNICUS/Landcover/100m/Proba-V/Global": {
        "0 Unknown": "282828",
        "20 Shrubs. Woody perennial plants with persistent and woody stems and without any defined main stem being less than 5 m tall. The shrub foliage can be either evergreen or deciduous.": "FFBB22",
        "30 Herbaceous vegetation. Plants without persistent stem or shoots above ground and lacking definite firm structure. Tree and shrub cover is less than 10 %.": "FFFF4C",
        "40 Cultivated and managed vegetation / agriculture. Lands covered with temporary crops followed by harvest and a bare soil period (e.g., single and multiple cropping systems). Note that perennial woody crops will be classified as the appropriate forest or shrub land cover type.": "F096FF",
        "50 Urban / built up. Land covered by buildings and other man-made structures.": "FA0000",
        "60 Bare / sparse vegetation. Lands with exposed soil, sand, or rocks and never has more than 10 % vegetated cover during any time of the year.": "B4B4B4",
        "70 Snow and ice. Lands under snow or ice cover throughout the year.": "F0F0F0",
        "80 Permanent water bodies. Lakes, reservoirs, and rivers. Can be either fresh or salt-water bodies.": "0032C8",
        "90 Herbaceous wetland. Lands with a permanent mixture of water and herbaceous or woody vegetation. The vegetation can be present in either salt, brackish, or fresh water.": "0096A0",
        "100 Moss and lichen.": "FAE6A0",
        "111 Closed forest, evergreen needle leaf. Tree canopy >70 %, almost all needle leaf trees remain green all year. Canopy is never without green foliage.": "58481F",
        "112 Closed forest, evergreen broad leaf. Tree canopy >70 %, almost all broadleaf trees remain green year round. Canopy is never without green foliage.": "009900",
        "113 Closed forest, deciduous needle leaf. Tree canopy >70 %, consists of seasonal needle leaf tree communities with an annual cycle of leaf-on and leaf-off periods.": "70663E",
        "114 Closed forest, deciduous broad leaf. Tree canopy >70 %, consists of seasonal broadleaf tree communities with an annual cycle of leaf-on and leaf-off periods.": "00CC00",
        "115 Closed forest, mixed.": "4E751F",
        "116 Closed forest, not matching any of the other definitions.": "007800",
        "121 Open forest, evergreen needle leaf. Top layer- trees 15-70 % and second layer- mixed of shrubs and grassland, almost all needle leaf trees remain green all year. Canopy is never without green foliage.": "666000",
        "122 Open forest, evergreen broad leaf. Top layer- trees 15-70 % and second layer- mixed of shrubs and grassland, almost all broadleaf trees remain green year round. Canopy is never without green foliage.": "8DB400",
        "123 Open forest, deciduous needle leaf. Top layer- trees 15-70 % and second layer- mixed of shrubs and grassland, consists of seasonal needle leaf tree communities with an annual cycle of leaf-on and leaf-off periods.": "8D7400",
        "124 Open forest, deciduous broad leaf. Top layer- trees 15-70 % and second layer- mixed of shrubs and grassland, consists of seasonal broadleaf tree communities with an annual cycle of leaf-on and leaf-off periods.": "A0DC00",
        "125 Open forest, mixed.": "929900",
        "126 Open forest, not matching any of the other definitions.": "648C00",
        "200 Oceans, seas. Can be either fresh or salt-water bodies.": "000080",
    },
    # USDA NASS Cropland Data Layers https://developers.google.com/earth-engine/datasets/catalog/USDA_NASS_CDL
    "USDA/NASS/CDL": {
        "1 Corn": "ffd300",
        "2 Cotton": "ff2626",
        "3 Rice": "00a8e2",
        "4 Sorghum": "ff9e0a",
        "5 Soybeans": "267000",
        "6 Sunflower": "ffff00",
        "10 Peanuts": "70a500",
        "11 Tobacco": "00af49",
        "12 Sweet Corn": "dda50a",
        "13 Pop or Orn Corn": "dda50a",
        "14 Mint": "7cd3ff",
        "21 Barley": "e2007c",
        "22 Durum Wheat": "896054",
        "23 Spring Wheat": "d8b56b",
        "24 Winter Wheat": "a57000",
        "25 Other Small Grains": "d69ebc",
        "26 Dbl Crop WinWht/Soybeans": "707000",
        "27 Rye": "aa007c",
        "28 Oats": "a05989",
        "29 Millet": "700049",
        "30 Speltz": "d69ebc",
        "31 Canola": "d1ff00",
        "32 Flaxseed": "7c99ff",
        "33 Safflower": "d6d600",
        "34 Rape Seed": "d1ff00",
        "35 Mustard": "00af49",
        "36 Alfalfa": "ffa5e2",
        "37 Other Hay/Non Alfalfa": "a5f28c",
        "38 Camelina": "00af49",
        "39 Buckwheat": "d69ebc",
        "41 Sugarbeets": "a800e2",
        "42 Dry Beans": "a50000",
        "43 Potatoes": "702600",
        "44 Other Crops": "00af49",
        "45 Sugarcane": "af7cff",
        "46 Sweet Potatoes": "702600",
        "47 Misc Vegs & Fruits": "ff6666",
        "48 Watermelons": "ff6666",
        "49 Onions": "ffcc66",
        "50 Cucumbers": "ff6666",
        "51 Chick Peas": "00af49",
        "52 Lentils": "00ddaf",
        "53 Peas": "54ff00",
        "54 Tomatoes": "f2a377",
        "55 Caneberries": "ff6666",
        "56 Hops": "00af49",
        "57 Herbs": "7cd3ff",
        "58 Clover/Wildflowers": "e8bfff",
        "59 Sod/Grass Seed": "afffdd",
        "60 Switchgrass": "00af49",
        "61 Fallow/Idle Cropland": "bfbf77",
        "63 Forest": "93cc93",
        "64 Shrubland": "c6d69e",
        "65 Barren": "ccbfa3",
        "66 Cherries": "ff00ff",
        "67 Peaches": "ff8eaa",
        "68 Apples": "ba004f",
        "69 Grapes": "704489",
        "70 Christmas Trees": "007777",
        "71 Other Tree Crops": "af9970",
        "72 Citrus": "ffff7c",
        "74 Pecans": "b5705b",
        "75 Almonds": "00a582",
        "76 Walnuts": "e8d6af",
        "77 Pears": "af9970",
        "81 Clouds/No Data": "f2f2f2",
        "82 Developed": "999999",
        "83 Water": "4970a3",
        "87 Wetlands": "7cafaf",
        "88 Nonag/Undefined": "e8ffbf",
        "92 Aquaculture": "00ffff",
        "111 Open Water": "4970a3",
        "112 Perennial Ice/Snow": "d3e2f9",
        "121 Developed/Open Space": "999999",
        "122 Developed/Low Intensity": "999999",
        "123 Developed/Med Intensity": "999999",
        "124 Developed/High Intensity": "999999",
        "131 Barren": "ccbfa3",
        "141 Deciduous Forest": "93cc93",
        "142 Evergreen Forest": "93cc93",
        "143 Mixed Forest": "93cc93",
        "152 Shrubland": "c6d69e",
        "176 Grassland/Pasture": "e8ffbf",
        "190 Woody Wetlands": "7cafaf",
        "195 Herbaceous Wetlands": "7cafaf",
        "204 Pistachios": "00ff8c",
        "205 Triticale": "d69ebc",
        "206 Carrots": "ff6666",
        "207 Asparagus": "ff6666",
        "208 Garlic": "ff6666",
        "209 Cantaloupes": "ff6666",
        "210 Prunes": "ff8eaa",
        "211 Olives": "334933",
        "212 Oranges": "e27026",
        "213 Honeydew Melons": "ff6666",
        "214 Broccoli": "ff6666",
        "216 Peppers": "ff6666",
        "217 Pomegranates": "af9970",
        "218 Nectarines": "ff8eaa",
        "219 Greens": "ff6666",
        "220 Plums": "ff8eaa",
        "221 Strawberries": "ff6666",
        "222 Squash": "ff6666",
        "223 Apricots": "ff8eaa",
        "224 Vetch": "00af49",
        "225 Dbl Crop WinWht/Corn": "ffd300",
        "226 Dbl Crop Oats/Corn": "ffd300",
        "227 Lettuce": "ff6666",
        "229 Pumpkins": "ff6666",
        "230 Dbl Crop Lettuce/Durum Wht": "896054",
        "231 Dbl Crop Lettuce/Cantaloupe": "ff6666",
        "232 Dbl Crop Lettuce/Cotton": "ff2626",
        "233 Dbl Crop Lettuce/Barley": "e2007c",
        "234 Dbl Crop Durum Wht/Sorghum": "ff9e0a",
        "235 Dbl Crop Barley/Sorghum": "ff9e0a",
        "236 Dbl Crop WinWht/Sorghum": "a57000",
        "237 Dbl Crop Barley/Corn": "ffd300",
        "238 Dbl Crop WinWht/Cotton": "a57000",
        "239 Dbl Crop Soybeans/Cotton": "267000",
        "240 Dbl Crop Soybeans/Oats": "267000",
        "241 Dbl Crop Corn/Soybeans": "ffd300",
        "242 Blueberries": "000099",
        "243 Cabbage": "ff6666",
        "244 Cauliflower": "ff6666",
        "245 Celery": "ff6666",
        "246 Radishes": "ff6666",
        "247 Turnips": "ff6666",
        "248 Eggplants": "ff6666",
        "249 Gourds": "ff6666",
        "250 Cranberries": "ff6666",
        "254 Dbl Crop Barley/Soybeans": "267000",
    },
    "ALOS_landforms": {
        "11 Peak/ridge (warm)": "141414",
        "12 Peak/ridge": "383838",
        "13 Peak/ridge (cool)": "808080",
        "14 Mountain/divide": "EBEB8F",
        "15 Cliff": "F7D311",
        "21 Upper slope (warm)": "AA0000",
        "22 Upper slope": "D89382",
        "23 Upper slope (cool)": "DDC9C9",
        "24 Upper slope (flat)": "DCCDCE",
        "31 Lower slope (warm)": "1C6330",
        "32 Lower slope": "68AA63",
        "33 Lower slope (cool)": "B5C98E",
        "34 Lower slope (flat)": "E1F0E5",
        "41 Valley": "a975ba",
        "42 Valley (narrow)": "6f198c",
    },
}


def ee_table_to_legend(in_table, out_file):
    """Converts an Earth Engine color table to a dictionary

    Args:
        in_table (str): The input file path (*.txt) to the Earth Engine color table.
        out_file (str): The output file path (*.txt) to the legend dictionary.
    """
    # pkg_dir = os.path.dirname(pkg_resources.resource_filename("geemap", "geemap.py"))
    # ee_legend_table = os.path.join(pkg_dir, "data/template/ee_legend_table.txt")

    if not os.path.exists(in_table):
        print("The class table does not exist.")

    out_file = os.path.abspath(out_file)
    if not os.path.exists(os.path.dirname(out_file)):
        os.makedirs(os.path.dirname(out_file))

    legend_dict = {}
    with open(in_table) as f:
        lines = f.readlines()
        for index, line in enumerate(lines):
            if index > 0:
                items = line.split("\t")
                items = [item.strip() for item in items]
                color = items[1]
                key = items[0] + " " + items[2]
                legend_dict[key] = color

    out_lines = []
    out_lines.append("{\n")

    for key in legend_dict.keys():
        line = "\t'{}': '{}',\n".format(key, legend_dict[key])
        out_lines.append(line)

    out_lines[-1] = out_lines[-1].rstrip()[:-1] + "\n"
    out_lines.append("}\n")

    with open(out_file, "w") as f:
        f.writelines(out_lines)
