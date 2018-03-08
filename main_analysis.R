library(readtext)
library(dplyr)
library(stringr)
library(text2vec) # NLP框架
library(jiebaR)   # 分词

# 预处理文本 -------------------------------------------------------------------

# 读入文本
raw_data <- readtext("data/*", encoding = "GB18030")

# 取出文本类别
yuliaoku <- raw_data %>% 
  mutate(
    category = str_split(doc_id, "-", simplify = TRUE)[, 1]
  )

# 划分训练集与测试集
set.seed(1024)
train_yuliaoku <- yuliaoku %>% 
  group_by(category) %>% 
  sample_frac(0.75)
test_yuliaoku <- setdiff(yuliaoku, train_yuliaoku)

# 建立分词器
token_worker <- worker(type = "tag", 
                       # 载入停用词
                       stop_word = "dict/stopwords.utf8.txt")
# 分词函数
tok_fun <- function(articles, require_words = NULL) {
  lapply(articles, function(x) {
    if (length(x) == 0) {
      return(character(0))
    }
    tokens <- token_worker[x]
    # 只取名词
    tokens <- tokens[names(tokens) == "n"]
    # 是否只保留需要的词
    if (!is.null(require_words)) {
      tokens <- tokens[tokens %in% require_words]
    }
    names(tokens) <- NULL
    return (tokens)
  })
}

# 建立token
train_token <- itoken(train_yuliaoku$text,
                      tokenizer = tok_fun,
                      ids = train_yuliaoku$doc_id,
                      progressbar = TRUE)
# 分词
vocab <- create_vocabulary(train_token)
# 排序
vocab <- arrange(vocab, -term_count)
# 观察分词结果
# View(vocab)
# 取词频最高的词，这里选取前2000
top_num <- 2000
require_words <- vocab$term[1:top_num]

# 建立带过滤的分词器
filter_tok_fun <- function(articles) {
  return (tok_fun(articles, require_words))
}
filter_train_token <- itoken(train_yuliaoku$text,
                             tokenizer = filter_tok_fun,
                             isd = train_yuliaoku$doc_id,
                             progressbar = TRUE)
filter_vocab <- create_vocabulary(filter_train_token)

# 生成document-term矩阵
train_dtm <- create_dtm(filter_train_token, vocab_vectorizer(filter_vocab))
train_mtx <- as.matrix(train_dtm)
# 处理成data.frame的形式，以方便建模
train_df <- data.frame(train_mtx)
# 将列名改为英文，方便处理
# 词与变量名的产出一个mapper，在测试集中缺词时，需要补上
col_name_mapper <-  paste0("w", 1:top_num)
names(col_name_mapper) <- colnames(train_mtx)
names(train_df) <- col_name_mapper[names(train_df)]
# 加上类别
train_df$category <- train_yuliaoku$category

# 建模 ----------------------------------------------------------------------

library(mlr)

task <- makeClassifTask(data = train_df, target = "category")
# 使用随机森林模型
lrn <- makeLearner("classif.randomForest", ntree = 1000)
# 预处理，增加标准化
lrn <- makePreprocWrapperCaret(lrn, ppc.center = TRUE, ppc.scale = TRUE)
# ！训练较慢
rf_model <- train(lrn, task)

# 模型评估 --------------------------------------------------------------------

# 处理测试集
filter_test_token <- itoken(test_yuliaoku$text,
                            tokenizer = filter_tok_fun,
                            isd = test_yuliaoku$doc_id,
                            progressbar = TRUE)
filter_test_vocab <- create_vocabulary(filter_test_token)
test_dtm <- create_dtm(filter_test_token, vocab_vectorizer(filter_test_vocab))
test_mtx <- as.matrix(test_dtm)

test_col_names <- colnames(test_mtx)

test_df <- data.frame(test_mtx)
names(test_df) <- col_name_mapper[names(test_df)]
diff_cols <- setdiff(names(col_name_mapper), test_col_names)
if (length(diff_cols) > 0) {
  # 空缺的补上0
  for (v_name in diff_cols) {
    c_name <- col_name_mapper[v_name]
    test_df[c_name] <- 0
  }
}

test_df$category <- test_yuliaoku$category

# 预测结果
prediction <- predict(rf_model, newdata = test_df)
caret::confusionMatrix(table(prediction$data))
