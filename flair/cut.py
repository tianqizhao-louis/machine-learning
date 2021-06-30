with open('conll_03/eng.train', 'r', encoding='utf-8') as f:
    lines = f.readlines()
    print(len(lines))

with open('conll_03/slice_1', 'w', encoding='utf-8') as r:
    for i in range(0, 55937):
        r.write(lines[i])
    r.write('\n')

with open('conll_03/slice_2', 'w', encoding='utf-8') as r:
    for i in range(55938, 109802):
        r.write(lines[i])
    r.write('\n')

with open('conll_03/slice_3', 'w', encoding='utf-8') as r:
    for i in range(109803, 164705):
        r.write(lines[i])
    r.write('\n')

with open('conll_03/slice_4', 'w', encoding='utf-8') as r:
    for i in range(164706, len(lines) - 1):
        r.write(lines[i])
    r.write('\n')


with open('conll_03/slice_1', 'r', encoding='utf-8') as f:
    slice_1 = f.readlines()
    print(len(slice_1))

with open('conll_03/slice_2', 'r', encoding='utf-8') as f:
    slice_2 = f.readlines()
    print(len(slice_2))

with open('conll_03/slice_3', 'r', encoding='utf-8') as f:
    slice_3 = f.readlines()
    print(len(slice_3))

with open('conll_03/slice_4', 'r', encoding='utf-8') as f:
    slice_4 = f.readlines()
    print(len(slice_4))