# ============================================================
# HILL CLIMBING - BEST IMPROVEMENT (com Repair e Convergência)
# Projeto: Intelligent Decision Support System for USA Stores
# Objetivos: O1, O2, O3
# ============================================================

source("funcao_auxiliar_otimizacao.R")  # carrega evaluate_plan e stores

# ------------------------------------------------------------
# PARÂMETROS GLOBAIS
# ------------------------------------------------------------
PR_VALUES  <- seq(0.00, 0.30, by = 0.05)   # valores discretos de PR
MAX_HR     <- 30                            # máximo de workers por dia (por tipo)
MAX_UNITS  <- 10000                         # restrição O2/O3: unidades totais

# Previsões para a semana seguinte (substituir por output do modelo de forecasting)
prev_list <- list(
  baltimore   = c(97,  61,  65,  71,  65,  89, 125),
  lancaster   = c(110, 70,  75,  80,  72,  95, 140),
  philadelphia= c(230, 144, 154, 168, 154, 211, 298),
  richmond    = c(180, 110, 120, 130, 115, 160, 210)
)

# ------------------------------------------------------------
# FUNÇÃO OBJECTIVO: calcula métricas a partir de um estado
# state = list(HR_X, HR_J, PR) para uma loja
# ------------------------------------------------------------
eval_state <- function(state, prev, store_params) {
  evaluate_plan(prev, state$HR_X, state$HR_J, state$PR,
                store_params, verbose = FALSE)
}

# ------------------------------------------------------------
# PERTURBAÇÃO DE SOLUÇÕES (para uma loja)
# Alterações percentuais com distribuição normal: mu=1, std=0.05
# Novo valor = valor_atual * fator, onde fator ~ N(1, 0.05)
#   HR_X, HR_J: arredondados ao inteiro mais próximo, clamp [0, MAX_HR]
#   PR: snap ao valor discreto mais próximo em PR_VALUES
# ------------------------------------------------------------
perturb_state <- function(state, n_neighbors = 20) {
  neighbors <- vector("list", n_neighbors)
  n_days    <- 7
  
  for (i in seq_len(n_neighbors)) {
    s <- state
    
    # --- HR_X: fator multiplicativo ~ N(1, 0.05) ---
    fatores_x   <- rnorm(n_days, mean = 1, sd = 0.05)
    new_hr_x    <- round(pmax(state$HR_X, 1L) * fatores_x)
    s$HR_X      <- pmin(pmax(new_hr_x, 0L), MAX_HR)
    
    # --- HR_J: fator multiplicativo ~ N(1, 0.05) ---
    fatores_j   <- rnorm(n_days, mean = 1, sd = 0.05)
    new_hr_j    <- round(pmax(state$HR_J, 1L) * fatores_j)
    s$HR_J      <- pmin(pmax(new_hr_j, 0L), MAX_HR)
    
    # --- PR: fator multiplicativo ~ N(1, 0.05), snap ao valor discreto mais próximo ---
    fatores_pr  <- rnorm(n_days, mean = 1, sd = 0.05)
    new_pr_cont <- state$PR * fatores_pr
    new_pr_cont <- pmin(pmax(new_pr_cont, min(PR_VALUES)), max(PR_VALUES))
    s$PR        <- PR_VALUES[sapply(new_pr_cont,
                                    function(v) which.min(abs(PR_VALUES - v)))]
    
    neighbors[[i]] <- s
  }
  return(neighbors)
}

# Mantém alias para compatibilidade com as funções O1/O2/O3
get_neighbors <- perturb_state

# ------------------------------------------------------------
# ESTADO INICIAL (pode ser ajustado ou aleatório)
# ------------------------------------------------------------
initial_state <- function() {
  list(
    HR_X = rep(5, 7),
    HR_J = rep(5, 7),
    PR   = rep(0.10, 7)
  )
}

# =============================
# FUNÇÃO DE REPAIR GLOBAL
# =============================
# Recebe o conjunto completo de estados de todas as lojas e avalia
# o total de unidades globalmente (cada loja com a sua própria prev).
# Se violar MAX_UNITS, reduz o PR de todas as lojas em 5% por iteração
# (até 15 tentativas), fazendo snap ao valor discreto mais próximo em
# PR_VALUES. Reduzir PR baixa as unidades vendidas sem eliminar workers,
# preservando melhor a estrutura da solução.
# Retorna a lista de estados corrigidos (ou original se já válida).
# =============================
repair_global <- function(states, all_stores, prev_list) {
  
  # Avaliação global correcta: cada loja usa a sua própria prev
  calc_total_units <- function(sts) {
    sum(sapply(names(all_stores), function(sname) {
      res <- evaluate_plan(prev_list[[sname]],
                           sts[[sname]]$HR_X,
                           sts[[sname]]$HR_J,
                           sts[[sname]]$PR,
                           all_stores[[sname]], verbose = FALSE)
      res$total_units
    }))
  }
  
  for (tentativa in 1:15) {
    total_units <- calc_total_units(states)
    if (total_units <= MAX_UNITS) break
    
    # Reduz PR de todas as lojas em 5%, com snap ao valor discreto mais próximo
    for (sname in names(all_stores)) {
      new_pr_cont <- states[[sname]]$PR * 0.95
      new_pr_cont <- pmin(pmax(new_pr_cont, min(PR_VALUES)), max(PR_VALUES))
      states[[sname]]$PR <- PR_VALUES[sapply(new_pr_cont,
                                             function(v) which.min(abs(PR_VALUES - v)))]
    }
  }
  
  return(states)
}

# ============================================================
# O1: Maximizar lucro semanal — independente por loja
# (sem restrição de unidades — repair não se aplica)
# ============================================================
hill_climbing_O1 <- function(store_name, max_iter = 500) {
  prev        <- prev_list[[store_name]]
  store_params <- stores[[store_name]]
  state       <- initial_state()
  best_result <- eval_state(state, prev, store_params)
  best_profit <- best_result$R_weekly
  
  # Histórico de convergência
  history <- data.frame(
    iteration = 0,
    profit = best_profit
  )
  
  cat(sprintf("\n[O1] %s | Lucro inicial: %d\n", store_name, best_profit))
  
  for (iter in 1:max_iter) {
    neighbors    <- get_neighbors(state)
    best_neighbor       <- NULL
    best_neighbor_profit <- best_profit
    
    for (nb in neighbors) {
      res <- eval_state(nb, prev, store_params)
      if (res$R_weekly > best_neighbor_profit) {
        best_neighbor        <- nb
        best_neighbor_profit <- res$R_weekly
      }
    }
    
    if (is.null(best_neighbor)) {
      cat(sprintf("  Convergiu na iteração %d | Lucro final: %d\n", iter, best_profit))
      break
    }
    
    state       <- best_neighbor
    best_profit <- best_neighbor_profit
    
    # Guarda histórico
    history <- rbind(history, data.frame(
      iteration = iter,
      profit = best_profit
    ))
  }
  
  cat(sprintf("  >> Plano final para %s:\n", store_name))
  final_result <- evaluate_plan(prev, state$HR_X, state$HR_J, state$PR,
                                store_params, verbose = TRUE)
  return(list(state = state, result = final_result, store = store_name, history = history))
}

# ============================================================
# O2: Maximizar lucro com restrição: total unidades ≤ 10.000
# Optimiza todas as lojas em conjunto — com REPAIR
# ============================================================

# Avalia lucro total de todas as lojas dado um conjunto de estados
eval_all_stores <- function(states) {
  total_profit <- 0
  total_units  <- 0
  results      <- list()
  for (sname in names(stores)) {
    res <- eval_state(states[[sname]], prev_list[[sname]], stores[[sname]])
    total_profit <- total_profit + res$R_weekly
    total_units  <- total_units  + res$total_units
    results[[sname]] <- res
  }
  list(profit = total_profit, units = total_units, results = results)
}

hill_climbing_O2 <- function(max_iter = 500) {
  states <- setNames(lapply(names(stores), function(s) initial_state()), names(stores))
  
  # Repair ao estado inicial só se necessário
  current <- eval_all_stores(states)
  if (current$units > MAX_UNITS) states <- repair_global(states, stores, prev_list)
  
  current <- eval_all_stores(states)
  best_profit <- if (current$units <= MAX_UNITS) current$profit else -Inf
  
  # Histórico de convergência
  history <- data.frame(
    iteration = 0,
    profit = current$profit,
    units = current$units
  )
  
  cat(sprintf("\n[O2] Lucro inicial: %d | Unidades totais: %d\n",
              current$profit, current$units))
  
  for (iter in 1:max_iter) {
    best_neighbor_profit <- best_profit
    best_neighbor_states <- NULL
    
    # Perturba uma loja de cada vez
    for (sname in names(stores)) {
      neighbors <- get_neighbors(states[[sname]])
      for (nb in neighbors) {
        new_states <- states
        new_states[[sname]] <- nb
        ev <- eval_all_stores(new_states)
        
        # Repair só se necessário
        if (ev$units > MAX_UNITS) {
          new_states <- repair_global(new_states, stores, prev_list)
          ev <- eval_all_stores(new_states)
        }
        
        if (ev$units <= MAX_UNITS && ev$profit > best_neighbor_profit) {
          best_neighbor_profit <- ev$profit
          best_neighbor_states <- new_states
        }
      }
    }
    
    if (is.null(best_neighbor_states)) {
      cat(sprintf("  Convergiu na iteração %d\n", iter))
      break
    }
    states      <- best_neighbor_states
    best_profit <- best_neighbor_profit
    
    # Guarda histórico
    final_eval <- eval_all_stores(states)
    history <- rbind(history, data.frame(
      iteration = iter,
      profit = final_eval$profit,
      units = final_eval$units
    ))
  }
  
  final <- eval_all_stores(states)
  cat(sprintf("  Lucro total: %d | Unidades totais: %d\n",
              final$profit, final$units))
  for (sname in names(stores)) {
    cat(sprintf("\n  >> Plano final para %s:\n", sname))
    evaluate_plan(prev_list[[sname]], states[[sname]]$HR_X, states[[sname]]$HR_J,
                  states[[sname]]$PR, stores[[sname]], verbose = TRUE)
  }
  return(list(states = states, final = final, history = history))
}

# ============================================================
# O3: Maximizar O2 e Minimizar HR total
# Função objetivo combinada: lucro - penalidade * total_HR
# — com REPAIR
# ============================================================
hill_climbing_O3 <- function(max_iter = 500, hr_penalty = 10) {
  
  score_O3 <- function(profit, units, total_hr) {
    if (units > MAX_UNITS) return(-Inf)
    profit - hr_penalty * total_hr
  }
  
  states <- setNames(lapply(names(stores), function(s) initial_state()), names(stores))
  
  # Repair ao estado inicial só se necessário
  current <- eval_all_stores(states)
  if (current$units > MAX_UNITS) states <- repair_global(states, stores, prev_list)
  
  current <- eval_all_stores(states)
  total_hr_init <- sum(sapply(names(stores), function(s) current$results[[s]]$total_HR))
  best_score <- score_O3(current$profit, current$units, total_hr_init)
  
  # Histórico de convergência
  history <- data.frame(
    iteration = 0,
    score = best_score,
    profit = current$profit,
    total_hr = total_hr_init,
    units = current$units
  )
  
  cat(sprintf("\n[O3] Score inicial: %.1f | Lucro: %d | HR total: %d | Unidades: %d\n",
              best_score, current$profit, total_hr_init, current$units))
  
  for (iter in 1:max_iter) {
    best_neighbor_score  <- best_score
    best_neighbor_states <- NULL
    
    for (sname in names(stores)) {
      neighbors <- get_neighbors(states[[sname]])
      for (nb in neighbors) {
        new_states <- states
        new_states[[sname]] <- nb
        ev <- eval_all_stores(new_states)
        
        # Repair só se necessário
        if (ev$units > MAX_UNITS) {
          new_states <- repair_global(new_states, stores, prev_list)
          ev <- eval_all_stores(new_states)
        }
        
        total_hr <- sum(sapply(names(stores), function(s) ev$results[[s]]$total_HR))
        sc <- score_O3(ev$profit, ev$units, total_hr)
        if (sc > best_neighbor_score) {
          best_neighbor_score  <- sc
          best_neighbor_states <- new_states
        }
      }
    }
    
    if (is.null(best_neighbor_states)) {
      cat(sprintf("  Convergiu na iteração %d\n", iter))
      break
    }
    states     <- best_neighbor_states
    best_score <- best_neighbor_score
    
    # Guarda histórico
    final_eval <- eval_all_stores(states)
    total_hr <- sum(sapply(names(stores), function(s) final_eval$results[[s]]$total_HR))
    history <- rbind(history, data.frame(
      iteration = iter,
      score = best_score,
      profit = final_eval$profit,
      total_hr = total_hr,
      units = final_eval$units
    ))
  }
  
  final    <- eval_all_stores(states)
  total_hr <- sum(sapply(names(stores), function(s) final$results[[s]]$total_HR))
  cat(sprintf("  Score final: %.1f | Lucro total: %d | HR total: %d | Unidades: %d\n",
              best_score, final$profit, total_hr, final$units))
  for (sname in names(stores)) {
    cat(sprintf("\n  >> Plano final para %s:\n", sname))
    evaluate_plan(prev_list[[sname]], states[[sname]]$HR_X, states[[sname]]$HR_J,
                  states[[sname]]$PR, stores[[sname]], verbose = TRUE)
  }
  return(list(states = states, final = final, total_hr = total_hr, history = history))
}

# ============================================================
# FUNÇÕES PARA PLOTAR CURVAS DE CONVERGÊNCIA
# ============================================================

# Plota convergência de O1 (todas as lojas num só gráfico)
plot_convergence_O1 <- function(results_O1, output_file = "convergence_O1.png") {
  png(output_file, width = 1000, height = 600, res = 100)
  
  # Define cores para cada loja
  colors <- c("baltimore" = "#E41A1C", "lancaster" = "#377EB8", 
              "philadelphia" = "#4DAF4A", "richmond" = "#984EA3")
  
  # Encontra limites do gráfico
  all_iters <- unlist(lapply(results_O1, function(r) r$history$iteration))
  all_profits <- unlist(lapply(results_O1, function(r) r$history$profit))
  
  plot(NULL, xlim = c(0, max(all_iters)), ylim = c(min(all_profits) * 0.95, max(all_profits) * 1.05),
       xlab = "Iteração", ylab = "Lucro Semanal",
       main = "Convergência O1: Maximizar Lucro (por loja)",
       cex.lab = 1.2, cex.main = 1.3)
  
  grid()
  
  # Plota cada loja
  for (store_name in names(results_O1)) {
    hist <- results_O1[[store_name]]$history
    lines(hist$iteration, hist$profit, col = colors[store_name], lwd = 2)
    points(hist$iteration, hist$profit, col = colors[store_name], pch = 19, cex = 0.5)
  }
  
  legend("bottomright", legend = names(results_O1), col = colors[names(results_O1)], 
         lwd = 2, pch = 19, cex = 1.1, bg = "white")
  
  dev.off()
  cat(sprintf("\n[PLOT] Curva de convergência O1 salva em: %s\n", output_file))
}

# Plota convergência de O2
plot_convergence_O2 <- function(result_O2, output_file = "convergence_O2.png") {
  png(output_file, width = 1000, height = 700, res = 100)
  
  par(mfrow = c(2, 1), mar = c(4, 4, 3, 2))
  
  hist <- result_O2$history
  
  # Gráfico 1: Lucro
  plot(hist$iteration, hist$profit, type = "o", col = "#377EB8", lwd = 2, pch = 19,
       xlab = "Iteração", ylab = "Lucro Total",
       main = "Convergência O2: Lucro Total (todas as lojas)",
       cex.lab = 1.2, cex.main = 1.3)
  grid()
  
  # Gráfico 2: Unidades
  plot(hist$iteration, hist$units, type = "o", col = "#E41A1C", lwd = 2, pch = 19,
       xlab = "Iteração", ylab = "Unidades Totais",
       main = sprintf("Unidades Totais (limite = %d)", MAX_UNITS),
       cex.lab = 1.2, cex.main = 1.3)
  abline(h = MAX_UNITS, col = "red", lty = 2, lwd = 2)
  grid()
  
  dev.off()
  cat(sprintf("\n[PLOT] Curva de convergência O2 salva em: %s\n", output_file))
}

# Plota convergência de O3
plot_convergence_O3 <- function(result_O3, output_file = "convergence_O3.png") {
  png(output_file, width = 1000, height = 1000, res = 100)
  
  par(mfrow = c(3, 1), mar = c(4, 4, 3, 2))
  
  hist <- result_O3$history
  
  # Gráfico 1: Score combinado
  plot(hist$iteration, hist$score, type = "o", col = "#4DAF4A", lwd = 2, pch = 19,
       xlab = "Iteração", ylab = "Score (Lucro - Penalidade*HR)",
       main = "Convergência O3: Score Combinado",
       cex.lab = 1.2, cex.main = 1.3)
  grid()
  
  # Gráfico 2: Lucro
  plot(hist$iteration, hist$profit, type = "o", col = "#377EB8", lwd = 2, pch = 19,
       xlab = "Iteração", ylab = "Lucro Total",
       main = "Lucro Total",
       cex.lab = 1.2, cex.main = 1.3)
  grid()
  
  # Gráfico 3: HR Total
  plot(hist$iteration, hist$total_hr, type = "o", col = "#984EA3", lwd = 2, pch = 19,
       xlab = "Iteração", ylab = "Total HR",
       main = "Total de Horas de Trabalho",
       cex.lab = 1.2, cex.main = 1.3)
  grid()
  
  dev.off()
  cat(sprintf("\n[PLOT] Curva de convergência O3 salva em: %s\n", output_file))
}

# Plota comparação entre objetivos
plot_comparison <- function(results_O1, result_O2, result_O3, output_file = "convergence_comparison.png") {
  png(output_file, width = 1200, height = 500, res = 100)
  
  par(mfrow = c(1, 3), mar = c(4, 4, 3, 2))
  
  # O1: Usa a soma dos lucros de todas as lojas
  o1_profits <- lapply(results_O1, function(r) r$history$profit)
  max_len <- max(sapply(o1_profits, length))
  
  # Preenche com último valor conhecido
  o1_profits_filled <- lapply(o1_profits, function(p) {
    if (length(p) < max_len) {
      c(p, rep(p[length(p)], max_len - length(p)))
    } else {
      p
    }
  })
  
  o1_total <- Reduce("+", o1_profits_filled)
  o1_iters <- 0:(length(o1_total) - 1)
  
  # Gráfico O1
  plot(o1_iters, o1_total, type = "o", col = "#E41A1C", lwd = 2, pch = 19,
       xlab = "Iteração", ylab = "Lucro Total",
       main = "O1: Sem Restrições",
       cex.lab = 1.2, cex.main = 1.3)
  grid()
  
  # Gráfico O2
  hist_o2 <- result_O2$history
  plot(hist_o2$iteration, hist_o2$profit, type = "o", col = "#377EB8", lwd = 2, pch = 19,
       xlab = "Iteração", ylab = "Lucro Total",
       main = "O2: Com Limite Unidades",
       cex.lab = 1.2, cex.main = 1.3)
  grid()
  
  # Gráfico O3
  hist_o3 <- result_O3$history
  plot(hist_o3$iteration, hist_o3$score, type = "o", col = "#4DAF4A", lwd = 2, pch = 19,
       xlab = "Iteração", ylab = "Score",
       main = "O3: Lucro + Min HR",
       cex.lab = 1.2, cex.main = 1.3)
  grid()
  
  dev.off()
  cat(sprintf("\n[PLOT] Comparação entre objetivos salva em: %s\n", output_file))
}

# ============================================================
# VALIDAÇÃO DA FUNÇÃO evaluate_plan
# Plano de teste definido no enunciado (slide 16) para Baltimore.
# Resultado esperado: week profit = 146
# ============================================================
cat("======================================================\n")
cat(" VALIDAÇÃO: resultado esperado = 146\n")
cat("======================================================\n")

prev_baltimore <- c(97, 61, 65, 71, 65, 89, 125)

HR_X_test <- c(4,  0, 8, 20, 0, 4, 3)
HR_J_test <- c(0, 10, 4,  0, 5, 5, 4)
PR_test   <- c(0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)

result_validation <- evaluate_plan(prev_baltimore, HR_X_test, HR_J_test, PR_test,
                                   stores$baltimore, verbose = TRUE)

if (result_validation$R_weekly == 146) {
  cat("\n[OK] Validacao passou: week profit = 146\n")
} else {
  cat(sprintf("\n[ERRO] Validacao falhou: week profit = %d (esperado 146)\n",
              result_validation$R_weekly))
}

# ============================================================
# EXECUÇÃO
# ============================================================
cat("\n======================================================\n")
cat(" HILL CLIMBING - BEST IMPROVEMENT (com Repair)\n")
cat("======================================================\n")

# --- O1: por loja (sem restrição — repair não aplicável) ---
cat("\n### OBJETIVO O1: Maximizar lucro (sem restrições) ###\n")
res_O1 <- lapply(names(stores), hill_climbing_O1)
names(res_O1) <- names(stores)

# --- O2: todas as lojas em conjunto (com repair) ---
cat("\n### OBJETIVO O2: Maximizar lucro (unidades totais <= 10.000) [com Repair] ###\n")
res_O2 <- hill_climbing_O2()

# --- O3: lucro + minimizar HR (com repair) ---
cat("\n### OBJETIVO O3: Maximizar O2 + Minimizar HR [com Repair] ###\n")
res_O3 <- hill_climbing_O3(hr_penalty = 10)

# ============================================================
# GERAR CURVAS DE CONVERGÊNCIA
# ============================================================
cat("\n======================================================\n")
cat(" GERANDO CURVAS DE CONVERGÊNCIA\n")
cat("======================================================\n")

plot_convergence_O1(res_O1)
plot_convergence_O2(res_O2)
plot_convergence_O3(res_O3)
plot_comparison(res_O1, res_O2, res_O3)

cat("\n[OK] Todas as curvas de convergência foram geradas!\n")
