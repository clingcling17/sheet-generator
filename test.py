import sys
import zipfile
from pathlib import Path
import pandas as pd
import numpy as np
from enum import Enum
from unittest import TestCase
from functools import total_ordering
from collections import defaultdict


class Columns(Enum):
     INFO_1_GENE_NAME = ('INFO.1.GENE_NAME', 'Gene_name')
     FUNC1_GENE_NAME = ('FUNC1.gene', 'Gene_name')
     AA_CHANGE = ('FUNC1.protein', 'AA_Change')
     NUCLEOTIDE_CHANGE  = ('FUNC1.coding', 'Nucleotide_change')
     VAF = ('INFO.A.AF', 'VAF')
     TOTAL_DEPTH = ('INFO.1.FDP', 'Total_depth')
     VARIANT_COUNT = ('INFO.A.FAO', 'Variant_count')
     REFSEQ = ('FUNC1.transcript', 'RefSeq')
     MUTATION_TYPE = ('FUNC1.function', 'mutation_type')
     ONCOMINE_GENE_CLASS = ('FUNC1.oncomineGeneClass', 'Oncomine_gene_class') #Function
     HOTSPOT = ('FUNC1.oncomineVariantClass', 'Hotspot')
     LOCATION = ('FUNC1.location', 'Location')
     ROWTYPE = ('rowtype', 'rowtype')
     COSM_ID = ('INFO...OID', 'COSM_ID')
     REFSNP_ID = ('FUNC1.CLNID1', 'RefSNP_id')
     REFSNP_STAT = ('FUNC1.CLNRENVSTAT1', 'RefSNP_stat')
     CLINICAL_SIGNIFICANCE = ('FUNC1.CLNSIG1', 'Clinical_significance')
     SIFT_SCORE = ('FUNC1.sift', 'SIFT_score')
     POLYPHEN_SCORE = ('FUNC1.polyphen', 'PolyPhen_score')
     GRANTHAM_SCORE = ('FUNC1.grantham', 'Grantham_score')
     FAIL_REASON = ('INFO.1.FAIL_REASON', 'Fail_reason')
     CHROMOSOME = ('CHROM', 'Chromosome')
     POSITION = ('POS', 'Position')
     END_POSITION = ('INFO.1.END', 'End_position')
     COPY_NUMBER = ('FORMAT.1.CN', 'Copy_number')
     CALL = ('call', 'Call')
     CI = ('INFO...CI', 'CI')
     ID = ('ID', 'ID')
     LENGTH = ('INFO...LEN', 'Length')
     QUALITY = ('QUAL', 'Quality')
     VCF_ROWNUM = ('vcf.rownum', 'vcf_rownum')
     MAPD = ('INFO...CDF_MAPD', 'MAPD')
     ALTERATION = ('ALT', 'Alteration')
     TOTAL_READ = ('INFO...READ_COUNT', 'Total_Read')
     EXON_NUMBER = ('INFO.1.EXON_NUM', 'Exon_number')
     ANNOTATION = ('INFO.1.ANNOTATION', 'Annotation')
     FILTER = ('FILTER', 'Filter')
     TIER = ('Tier', 'Tier')
    

     def __init__(self, tsv, readable):
        self._value_ = tsv 
        self.readable = readable

     def __str__(self):
         return self.value
     
     @staticmethod
     def getReadableName(tsv):
         for column in Columns:
             if column.value == tsv:
                 return column.readable
         return tsv


Col = Columns



@total_ordering
class Tiers(Enum): 
    TIER_1_2 = 'I/II'
    TIER_3_4 = 'III/IV'
    TIER_4 = 'IV'

    @property
    def index(self):
        return list(Tiers).index(self)
    
    def __str__(self):
        return self.value

    def __lt__(self, other):
        if self.__class__ is other.__class__:
            return self.index < other.index


assert(Tiers.TIER_1_2 < Tiers.TIER_3_4)



class FileProcessor:
    def __init__(self, file_path, dest_dir_path):
        self.file_path = Path(file_path)
        self.case_name = self.file_path.stem.split('_')[0]
        self.dest_dir = Path(dest_dir_path, self.case_name)
    

    def unzip_to_destination_and_normalize(self):
        # 목표 디렉토리가 비어있는지 검사
        if self.dest_dir.exists():
            sys.exit('The specified directory %s already exists.' % self.dest_dir)
        
        self.dest_dir.mkdir(parents=True, exist_ok=True)        

        zipdata = zipfile.ZipFile(self.file_path)
        zipinfos = zipdata.infolist()
        
        for zipinfo in zipinfos:
            zipinfo.filename = zipinfo.filename.replace(':', '-')
            zipdata.extract(zipinfo, self.dest_dir)
        
        return self.dest_dir


    def find_oncomine_file(self):
        path = Path(self.dest_dir)
        path = path / 'Variants'
        case_dirs = [x for x in path.iterdir() 
                     if x.is_dir and x.name.startswith(self.case_name)]
        if not case_dirs:
            sys.exit('There is no case directory in Variants directory.')
        targets = [x for x in case_dirs[0].iterdir() 
                   if x.is_file and x.name.endswith('-oncomine.tsv')]
                   
        if not targets:
            sys.exit('There is no -oncomine.tsv file in %s.' % path)
        return targets[0]



class OncomineParser:

    def __init__(self, oncomine_file):
        self.oncomine_file = oncomine_file
        

    ## https://pandas.pydata.org/docs/user_guide/basics.html#basics-dtypes

    def parse_oncomine_file(self):
        def def_type():
            return pd.StringDtype

        dtype = defaultdict(def_type())
        dtype[Col.VAF.value] = pd.Float32Dtype
        dtype[Col.TOTAL_DEPTH.value] = pd.UInt16Dtype
        dtype[Col.POSITION.value] = pd.UInt32Dtype
        dtype[Col.END_POSITION.value] = pd.UInt32Dtype
        dtype[Col.LENGTH.value] = pd.UInt32Dtype
        dtype[Col.EXON_NUMBER.value] = pd.UInt8Dtype
        dtype[Col.COPY_NUMBER.value] = pd.UInt8Dtype
        dtype[Col.QUALITY.value] = pd.Float32Dtype
    
        return pd.read_table(self.oncomine_file, index_col='vcf.rownum',
                             dtype=dtype, na_values = ['.'], comment='#',
                             low_memory=False)


class ReportGenerator:
    
    def __init__(self, dataframe):
        self.df = dataframe
    

    def generate_report(self):
        snv = self.generate_snv()
        snv_nocall = self.generate_snv_nocall()
        cnv = self.generate_cnv()
        cnv_nocall = self.generate_cnv_nocall()
        fusion = self.generate_fusion()
        fusion_nocall = self.generate_fusion_nocall()

        reports = {
            'SNV': snv,
            'SNV_NOCALL': snv_nocall,
            'CNV': cnv,
            'CNV_NOCALL': cnv_nocall,
            'Fusion': fusion,
            'FUSION_NOCALL': fusion_nocall
        }
        
        return reports
        

    # https://www.statology.org/pandas-rename-columns-with-dictionary/
        
    def generate_snv(self):
        columns = [
            Col.FUNC1_GENE_NAME, Col.NUCLEOTIDE_CHANGE, Col.AA_CHANGE, 
            Col.TOTAL_DEPTH, Col.VAF, Col.VARIANT_COUNT, Col.REFSEQ, Col.MUTATION_TYPE, 
            Col.ONCOMINE_GENE_CLASS, Col.HOTSPOT, Col.LOCATION, Col.ROWTYPE, 
            Col.COSM_ID, Col.REFSNP_ID, Col.REFSNP_STAT, 
            Col.CLINICAL_SIGNIFICANCE, Col.SIFT_SCORE, Col.POLYPHEN_SCORE, 
            Col.GRANTHAM_SCORE, Col.FAIL_REASON
        ]
        condition = f'`{Col.CALL}` == "POS"'\
            f' and `{Col.LOCATION}` not in ["intronic", "utr_3", "utr_5"]'\
            f' and `{Col.ROWTYPE}` in ["snp", "del", "ins", "complex", "mnp", "RNAExonTiles"]'\
            f' and `{Col.MUTATION_TYPE}`.notna()'\
            f' and `{Col.MUTATION_TYPE}` != "synonymous"'
        # tmb_gene not in 조건은 일단 생략함.

        snv.insert(4, Col.TIER.value, np.nan)
        snv = self.generate_filtered_dataframe(condition, columns)
        if not snv.empty:
            snv[Col.TIER.value] = snv.apply(self.populate_tier_default, axis=1)
            snv.loc[snv[Col.TOTAL_DEPTH.value] < 100, Col.TIER.value] = Tiers.TIER_4
        return snv
    

    def generate_snv_nocall(self):
        columns = [
            Col.FUNC1_GENE_NAME, Col.REFSEQ, Col.MUTATION_TYPE, Col.AA_CHANGE,
            Col.NUCLEOTIDE_CHANGE, Col.VAF, Col.TOTAL_DEPTH, Col.VARIANT_COUNT,
            Col.ONCOMINE_GENE_CLASS, Col.LOCATION, Col.ROWTYPE, Col.COSM_ID,
            Col.REFSNP_ID, Col.REFSNP_STAT, Col.CLINICAL_SIGNIFICANCE,
            Col.FAIL_REASON
        ]
        condition = f'`{Col.CALL}` == "NOCALL"'\
            f' and `{Col.ROWTYPE}` not in ["CNV", "Fusion"]'
        
        return self.generate_filtered_dataframe(condition, columns)
    

    def generate_cnv(self):
        columns = [
            Col.FUNC1_GENE_NAME, Col.COPY_NUMBER, Col.CHROMOSOME, Col.POSITION, 
            Col.END_POSITION, Col.CALL, Col.CI, Col.FILTER, Col.ROWTYPE, Col.ID,
            Col.LENGTH, Col.ONCOMINE_GENE_CLASS, Col.HOTSPOT, Col.QUALITY, 
            Col.VCF_ROWNUM, Col.MAPD, Col.CLINICAL_SIGNIFICANCE, Col.FAIL_REASON
        ]
        
        condition = f'`{Col.CALL}` in ["DEL", "AMP"]'\
            f' or `{Col.ROWTYPE}` == "LOH"'
        tier_2_3_gene_names = [
            'AKT1', 'ALK', 'BRAF', 'CCND2', 'CCNE1', 'CD274', 'CDK4', 'CDK6', 
            'DDR1', 'DDR2', 'EGFR', 'EMSY', 'ERBB2', 'FGF23', 'FGF3', 'FGF4', 
            'FGF19', 'CCND1', 'FGF9', 'FGFR1', 'FGFR2', 'FGFR4', 'GNAS', 'KRAS',
            'MAP2K1', 'MCL1', 'MDM2', 'MET', 'MYC', 'NTRK1', 'PIK3CA', 'PTPN11',
            'RAF1', 'RICTOR', 'ROS1', 'SRC'
        ]
        
        cnv = self.generate_filtered_dataframe(condition, columns)
        cnv.insert(2, Col.TIER.value, np.nan)
        if not cnv.empty:
            cnv[Col.TIER.value] = cnv.apply(self.populate_tier_default, axis=1)
            cnv.loc[(cnv[Col.ROWTYPE.value] == 'AMP') & (cnv[Col.FUNC1_GENE_NAME.value].isin(tier_2_3_gene_names)),
                      Col.TIER.value] = Tiers.TIER_1_2
        return cnv
    

    def generate_cnv_nocall(self):
        columns = [
            Col.FUNC1_GENE_NAME, Col.CALL, Col.COPY_NUMBER, Col.CI, Col.FILTER,
            Col.ROWTYPE, Col.ID, Col.CHROMOSOME, Col.POSITION, Col.LENGTH,
            Col.END_POSITION, Col.ONCOMINE_GENE_CLASS, Col.QUALITY, #vcf_rownum은 자동출력됨
            Col.MAPD, Col.FAIL_REASON
        ]
        condition = f'`{Col.CALL}` == "NOCALL"'\
            f' and `{Col.ROWTYPE}` in ["CNV", "LOH"]'
        
        return self.generate_filtered_dataframe(condition, columns)
    
    
    def generate_fusion(self):
        columns = [
            Col.FILTER, Col.CALL, Col.ROWTYPE, Col.ID, Col.CHROMOSOME, 
            Col.INFO_1_GENE_NAME, Col.ALTERATION, Col.POSITION, Col.TOTAL_READ,
            Col.ANNOTATION, Col.EXON_NUMBER, Col.VCF_ROWNUM, 
            Col.CLINICAL_SIGNIFICANCE, Col.HOTSPOT,Col.FAIL_REASON
        ]
        condition = f'`{Col.CALL}` == "POS"'\
            f' and `{Col.ROWTYPE}` in ["Fusion", "RNAExonVariant"]'
        tier_2_3_gene_names = (
            'ALK', 'BRAF', 'MET', 'ESR1', 'EGFR', 'ETV6', 'NTRK3', 'FLI1',
            'FGFR', 'FGFR3', 'NTRK2', 'NRG1', 'NTRK3', 'PAX8', 'RAF1', 'RELA',
            'RET', 'PIK3CA'
        )
        
        fusion = self.generate_filtered_dataframe(condition, columns)
        if not fusion.empty:
            fusion[Col.TIER.value] = fusion.apply(self.populate_tier_default, axis=1)
            fusion.loc[fusion[Col.INFO_1_GENE_NAME.value]
                       .str.startswith(tier_2_3_gene_names), Col.TIER.value] = Tiers.TIER_1_2
        return fusion
    

    def generate_fusion_nocall(self):
        columns = [
            Col.FILTER, Col.CALL, Col.ROWTYPE, Col.ID, Col.CHROMOSOME, 
            Col.INFO_1_GENE_NAME, Col.ALTERATION, Col.POSITION, Col.TOTAL_READ,
            Col.ANNOTATION, Col.EXON_NUMBER, Col.ONCOMINE_GENE_CLASS,
            Col.FAIL_REASON
        ]
        condition = f'{Col.FILTER} == "FAIL"'\
            f' and `{Col.ROWTYPE}` in ["Fusion", "RNAExonVariant"]'
        
        return self.generate_filtered_dataframe(condition, columns)
    

    # omit nonexistent column
    def generate_filtered_dataframe(self, condition_expr, columns):
        column_names = map(lambda x: x.value, columns)
        return self.df.query(condition_expr)[self.df.columns.intersection(column_names)]
    

    def populate_tier_default(self, row):
        clinical_significance = row[Col.CLINICAL_SIGNIFICANCE.value]
        hotspot = row[Col.HOTSPOT.value]

        tier = ''
        if pd.notna(clinical_significance):
            if 'benign' in clinical_significance.lower():
                tier = Tiers.TIER_4
            elif clinical_significance == 'not_provided'\
                or clinical_significance == 'Uncertain_significance'\
                or 'conflicting' in clinical_significance.lower():
                tier = Tiers.TIER_3_4
        if pd.notna(hotspot):
            if hotspot == 'Deleterious' or hotspot == 'Hotspot':
                tier = Tiers.TIER_1_2

        return tier        
    


# TODO file existence check
class ExcelWriter:    
    def __init__(self, dataframes: dict, file: Path):
        self.dataframes = dataframes
        self.file = file

    def write(self):
        self.file.parent.mkdir(parents=True)
        with pd.ExcelWriter(self.file, engine='xlsxwriter') as writer:
            for key in self.dataframes:
                df = self.dataframes[key].rename(Col.getReadableName, axis='columns')
                df.to_excel(writer, sheet_name = key)



class Tests(TestCase):
    def test_parser(self):
        parser = OncomineParser('M23-6180_v1_M23-6180_RNA_v1_Non-Filtered_2023-08-07_21-01-24-oncomine.tsv')
        parser.parse_oncomine_file()

def main():
    if len(sys.argv) != 3:
        sys.exit('Please check arguments number.\n'\
                 + 'Usage: run.exe Mxx-xxxx.zip /destination/directory')
    file_path = sys.argv[1]
    dest_path = sys.argv[2]
    print('File path: ' + file_path)
    print('Destination path: ' + dest_path)

    fileProcessor = FileProcessor(file_path, dest_path)
    case_name = fileProcessor.case_name
    dest_path = fileProcessor.unzip_to_destination_and_normalize()
    oncomine_file = fileProcessor.find_oncomine_file()

    parser = OncomineParser(oncomine_file)
    dataframe = parser.parse_oncomine_file()
    reportGenerator = ReportGenerator(dataframe)
    reports = reportGenerator.generate_report()

    file = Path(dest_path, 'result', case_name + '.xlsx')
    writer = ExcelWriter(reports, file)
    writer.write()
    print('Generated report worksheet: ' + str(file))

    # parser = OncomineParser('M23-6180_v1_M23-6180_RNA_v1_Non-Filtered_2023-08-07_21-01-24-oncomine.tsv')
    # dataframe = parser.parse_oncomine_file()
    # reportGen = ReportGenerator(dataframe)
    # filtered_df = reportGen.generate_snv()
    # filtered_df.rename(Col.getReadableName, axis='columns', inplace=True)
    # print(filtered_df)


class ReportTextGenerator():

    def __init__(self, snv, cnv, fusion, file):
        self.snv = snv
        self.cnv = cnv
        self.fusion = fusion
        self.file = file

    def print(self):
        snv = self.snv
        cnv = self.cnv
        fusion = self.fusion

        tier_1_2_mutation = snv.loc[snv[Col.TIER.value] == Tiers.TIER_1_2][
            Col.FUNC1_GENE_NAME.value, Col.AA_CHANGE.value, 
            Col.NUCLEOTIDE_CHANGE.value, Col.TIER.value
        ]
        
        tier_1_2_amplification = cnv.loc[cnv[Col.TIER.value] == Tiers.TIER_1_2][
            Col.FUNC1_GENE_NAME.value, Col.COPY_NUMBER.value, Col.TIER.value
        ]
        
        # tier_1_2_fusion = fusion.loc[fusion[Col.TIER.value] == 'I/II'][
        #     Col.INFO_1_GENE_NAME.value, Col.CHROMOSOME.value, Col.GENE
        # ]


    def printMutation(self):
        f = self.file
        f.write('(1)Mutation\n')
        self.snv.to_csv(f, sep = '\t', index = False, columns = [
            Col.FUNC1_GENE_NAME.value, Col.AA_CHANGE.value, 
            Col.NUCLEOTIDE_CHANGE.value, Col.VAF.value, Col.TIER.value
            ])
        f.write('\n')

if __name__ == "__main__":
    main()


