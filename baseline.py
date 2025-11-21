import pandas as pd
from pathlib import Path

'''
Used for dataloader2.py, the site to process parameters
'''
def gen_site_csv(site = None):
        if site==None:
                site_ids = [
                        "942030-99999",
                        "943350-99999",
                        "943740-99999",
                        "944760-99999",
                        "474250-99999",
                        "477590-99999",
                        "477710-99999",
                        "478030-99999",
                        "471100-99999",
                        "471080-99999",
                        "471420-99999",
                        "471510-99999",
                        "760500-99999",
                        "760610-99999",
                        "761130-99999",
                        "761220-99999",
                        "843900-99999",
                        "844520-99999",
                        "846910-99999",
                        "847520-99999",
                        "726810-24131",
                        "726815-24106",
                        "722860-23119",
                        "722265-13821",
                ]

                rows = []
                for sid in site_ids:
                    usaf, wban = sid.split("-")
                    rows.append({"USAF": usaf, "WBAN": wban})

                df = pd.DataFrame(rows)
                df.to_csv("sites_to_process.csv", index=False)

        else:
                print("check gen_site_csv() for supporting passing parameter")
                exit(1)








if __name__ == "__main__":
    gen_site_csv()