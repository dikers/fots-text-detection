import os
import glob
keys = '台融“不名联季期提能球信放高分政加满常席撑全正率退未证按轨将警主用记连决赛州琦表在购健华委部结前导胜布构更许同病真观击场持个顽察深度促运基就忆保备至复帮企据续新还第贷押康美控3停各需注定毒蔓到最顿告策比发衰调与方公情光膨并利滑。上创半供取疫生护例，说延水进买现而产缘抵断险和天泛承身挑（ 支体对降宽过通面短时件二债环储队得松诺卫盛出为景者超效估众拼平手应境经家济多总影其锋物力机国维活德%确耳助相也让意次响作鲍那）来胀冲工我会行界以0宣抱失大们快零他一妨今强9邦”8示流算继亿采数束.价世缓很款威成闻预程措元日东刻战了区道社减标下当广际商位此动业施地尔显类回府级因住危心电艰具达实年积是内已的打万值事攀周、目近明兴严2稳中共庭速摆赢非制金激受低首幅聪致碍使硬额计务如重4声5增所风于给后难向土币市间人直员可有券苏输月传冠货充0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZÉ-~`´<>\'.:;^/|!?$%#@&*()[]{}_+=,\\\"'





def get_key_from_file_list(input_dir):
    file_list = glob.glob(os.path.join(input_dir, '*.txt'))

    print(file_list)

    content = ''
    for file_name in file_list:
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                content += line

    content = content.replace('\n', '')
    content = set(content)
    content = ''.join(content)
    print(content)

#get_key_from_file_list('/Users/dikers/work/workspace/tfc/ocr_synth_text_chinese/synth_image/output/ch4_training_localization_transcription_gt')
