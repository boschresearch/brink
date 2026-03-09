#!/bin/bash

# -----------------------------------------------------
# 合并 AMIE 输出转换和全局索引分配
# 一步完成：txt -> json -> 带索引的json
# -----------------------------------------------------

set -e

# 配置参数 - 根据需要修改
DATASET="family"  # 或 "fb15k-237"
MAXAD="3"         # 或 "4"

# 自动构建路径
BASE_DIR="data/${DATASET}"
AMIE_TXT_FILE="amie_output_family.txt"
TEMP_JSON_FILE="${BASE_DIR}/rules_string.json"
FINAL_JSON_FILE="${BASE_DIR}/rules_with_global_index.json"

echo "==========================================="
echo "Converting and indexing AMIE rules"
echo "Dataset: ${DATASET}"
echo "Max atoms: ${MAXAD}"
echo "-------------------------------------------"

# 步骤1: 转换 AMIE 输出为 JSON
echo "🔄 Step 1: Converting AMIE txt to JSON..."
python convert_amie_txt_to_json.py "${AMIE_TXT_FILE}" "${TEMP_JSON_FILE}"

# 步骤2: 分配全局索引
echo "🔄 Step 2: Assigning global rule indices..."
python utils/assign_global_rule_index.py \
  -i "${TEMP_JSON_FILE}" \
  -o "${FINAL_JSON_FILE}"

# 步骤3: 清理临时文件（可选）
echo "🧹 Step 3: Cleaning up temporary files..."
rm "${TEMP_JSON_FILE}"

echo "-------------------------------------------"
echo "✅ Process completed successfully!"
echo "📁 Final output: ${FINAL_JSON_FILE}"
echo "🎯 Ready for rule filtering and grounding generation"
echo "==========================================="