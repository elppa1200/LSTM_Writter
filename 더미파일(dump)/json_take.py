import os, json

#document paragraph form
#def jsonfile():

path = 'dataset/NIKL_WRITTEN'
json_path = os.listdir(path)
json_return = []
i = 0
def jsonfile():

    for i in range(100):
        json_filename = json_path[i]
        json_filepath = 'dataset/NIKL_WRITTEN/' + json_filename

        with open(json_filepath,'r', encoding='utf-8') as json_file:
            json_data = json.load(json_file)
        json_docu = json_data['document']
        json_para = json_docu[0]['paragraph']

        for k in range(len(json_para)):
            json_form = json_para[k]['form']
            k = k + 1
            json_return.append(json_form)

        i = i + 1
        print('Json process : '+ str(i)+ '/ 100')

    print('Turn Json into string...')
    json_return_text = ' '.join(json_return)
    print('Json done')
    print('len :' + str(len(json_return_text)))#2924693076
    return(json_return_text)#str

