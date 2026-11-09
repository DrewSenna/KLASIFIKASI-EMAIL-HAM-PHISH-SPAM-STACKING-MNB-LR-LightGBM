#!/usr/bin/env bash
# usage: bash check_bahasa.sh Laporan_UTS_Group_1.md   (line numbers match the file)
F="${1:-Laporan_UTS_Group_1.md}"
# blank out inline code, image lines and link targets so code and paths do not trigger
T=$(mktemp); sed -E '/^## (Daftar Pustaka|Referensi)/,$d; s/`[^`]*`/CODE/g; s/^!\[.*$//; s/\]\([^)]*\)/]/g' "$F" > "$T"
g(){ printf '== %s\n' "$1"; grep -n -o -P "$2" "$T"; }
g 'Non-baku (KBBI) and relative di mana' '(?i)\b(analisa|menganalisa|resiko|praktek|obyek|sistim|metoda|kwalitas|hipotesa|diagnosa|teoritis|ekstrim|prosentase|presentase|sekedar|merubah|mempengaruhi|memerhatikan|mengkonversi|eksperimentasi|hirarki|aritmatika|pararel|sistimatis|subyek|mengkategori\w*|katagori|menterjemahkan|mengelasifikasi\w*|memroses|aktifitas|efektifitas|efektip|tehnik|frekwensi|konkrit|komplek|matrik|linier|algoritme|standard|apapun|kerjasama|bekerjasama|tanggungjawab|didalam|diantara|dimana|di mana|yang mana|adalah merupakan)\b'
g 'Comma before subordinator after main clause' '(?<!Namun|itu|demikian|lain|Misalnya|Artinya)(?<!Akan tetapi), (sehingga|karena|agar|supaya|jika|apabila|kalau|bila)\b'
g 'Sentence-initial conjunction' '(^|[.!?] )\**(Tetapi|Sehingga|Dan|Serta|Atau|Maka)\b'
g 'Bound form apart / lowercase hyphen' '\b(antar|non|pasca|pra|multi|sub|semi|anti) (?=[a-z*])|\b(antar|non|pasca|pra|multi|sub|semi|anti)-(?=[a-z])'
g 'Affix + foreign word' '(?i)\bdi ?(training|testing|tuning|tune|train|test|fit|resample|deploy)\w*|\bdi-(?!\*)\w+'
g 'Roman decided foreign word' '(?<![*\w-])(email|Email|training|Training|testing|Testing|fold|threshold|stopword)(?![*\w-])'
g 'Number format' '[0-9] %|[0-9] [–-] ?[0-9a-z]|[0-9] ?– [0-9a-z]|(?<![\w.])[0-9]+,[0-9]+-[0-9]|\b0\.[0-9]+\b'
g 'Sentence starts with a digit' '(?<=[.!?] )[0-9][0-9.,]*(?![.)] )'
g 'Enumerator followed by period' '(^\s*|[:;] )[0-9a-z]\)\.'
g 'APA leftovers' 'et al\.|\([A-Z][^()]{1,60}, (19|20)[0-9]{2}\)|[A-Z]\w+ \((19|20)[0-9]{2}\)'
g 'Caption not "Gambar x.y Judul"' '^\*?(Gambar|Tabel) [0-9]+[a-z]?(\.(?![0-9])|(?![.0-9]))'
g 'Ratio with spaces' '[0-9A-Za-z] : [0-9]'
rm -f "$T"
