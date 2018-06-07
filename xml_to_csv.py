import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        Mandatory=['LINE','DISCOUNT','PO_NUMBER','EFFECTIVE_START_DATE','QUOTE_ID','ITEM_NAME','NET_PRICE','PAYMENT_TERM','ORDER_TOTAL','EFFECTIVE_END_DATE','PART_NUMBER','LIST_PRICE','PO_DATE','DEAL_ID','UOM','QTY','LOGO','END_CUSTOMER_ADDRESS','ITEM_DESC','SUPPLIER','LINE_NOTES','BILL_TO_ADDRESS','CONTACT_INFORMATION',
        'NOTES','LINE_REF','SHIP_TO_ADDRESS','PO_TYPE','CURRENCY_CODE','PO_NUMBER','FOB','SHIPPING_NOTES','FREIGHT_CHARGES']
        Mandatory1=['PO_NUMBER']
        for member in root.findall('object'):
            #print(member[0].text) 
            if (member[0].text) in Mandatory:  
                #print(member[0].text)            
                value = (root.find('filename').text,
                         int(root.find('size')[0].text),
                         int(root.find('size')[1].text),
                         member[0].text,
                         int(member[4][0].text),
                         int(member[4][1].text),
                         int(member[4][2].text),
                         int(member[4][3].text)
                         )
                xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    for folder in ['train','test']:
        image_path = os.path.join(os.getcwd(), ('images/' + folder))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv(('images/' + folder + '_labels.csv'), index=None)
        print('Successfully converted xml to csv.')


main()
