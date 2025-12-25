import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter
import numpy as np
import io
import math
from scipy.optimize import curve_fit
# import japanize_matplotlib
import os

# 日本語フォント設定 (Streamlit Cloud w/ packages.txt or Local)
# IPAexGothicがインストールされているか確認し、あれば設定する
try:
    plt.rcParams['font.family'] = 'IPAexGothic'
except:
    pass

# ==============================================================================
# 0. 定数・設定
# ==============================================================================
TITLE = "散布図作成"
ICON = "📈"

# ==============================================================================
# 1. 計算ロジック
# ==============================================================================
def get_optimal_scale(max_val, use_si=True):
    """
    最大値に応じて最適なスケール倍率と単位サフィックスを返す関数
    """
    if max_val == 0 or pd.isna(max_val) or np.isinf(max_val):
        return 1.0, ""
        
    abs_val = abs(max_val)
    if abs_val == 0: return 1.0, ""

    log_val = math.log10(abs_val)
    
    if use_si:
        # SI接頭辞 (3の倍数) に合わせる
        exponent = math.floor(log_val / 3) * 3
        if exponent == 0: return 1.0, ""
        
        scale_factor = 10 ** (-exponent)
        prefixes = {
            -24: 'y', -21: 'z', -18: 'a', -15: 'f', -12: 'p', -9: 'n', -6: 'µ', -3: 'm',
            0: '',
            3: 'k', 6: 'M', 9: 'G', 12: 'T', 15: 'P', 18: 'E', 21: 'Z', 24: 'Y'
        }
        prefix = prefixes.get(exponent, None)
        if prefix is not None:
            return scale_factor, prefix # 括弧なしの純粋な接頭辞を返す
        else:
            return scale_factor, f"10^{{{exponent}}}"
    else:
        return 1.0, ""

def force_sci_format_func(x, pos):
    """
    強制的にすべての値を指数表記にする formatter
    """
    if x == 0: return "0"
    exponent = int(math.floor(math.log10(abs(x))))
    mantissa = x / (10**exponent)
    return fr"${mantissa:.2f} \times 10^{{{exponent}}}$"

def fmt_latex_num(val, precision=2):
    """
    数値をLaTeX形式の文字列に変換する (e表記を排除し 10^n にする)
    """
    if val == 0: return "0"
    
    exponent = int(math.floor(math.log10(abs(val))))
    
    # 指数が -3 ～ 3 の間なら通常の小数表記にする
    if -3 <= exponent <= 3:
        return f"{val:.{precision+1}g}"
    else:
        mantissa = val / (10**exponent)
        return fr"{mantissa:.{precision}f} \times 10^{{{exponent}}}"

def calc_r2(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    mask = ~np.isnan(y_true) & ~np.isnan(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]
    if len(y_true) < 2: return np.nan
    residuals = y_true - y_pred
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    if ss_tot == 0: return np.nan 
    return 1 - (ss_res / ss_tot)

def model_linear(x, a, b): return a * x + b
def model_poly(x, *coeffs): return np.polyval(coeffs, x)
def model_exp(x, a, b, c): return a * np.exp(b * x) + c 

# ==============================================================================
# 2. メインアプリ
# ==============================================================================
def main():
    st.set_page_config(page_title=TITLE, layout="wide", page_icon=ICON)
    
    st.markdown(f"## {ICON} {TITLE}")
    
    # Layout: 1. Data | 2. Settings | 3. Result
    col_data, col_settings, col_result = st.columns([1, 1.2, 2.0])
    
    # Shared Data Container
    df = None
    all_series_config = [] 
    
    # ---------------------------------------------------------
    # Column 1: Data
    # ---------------------------------------------------------
    with col_data:
        st.header("1. Data")
        uploaded_file = st.file_uploader("CSV / Excel Upload", type=["csv", "xlsx"])
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_names = excel_file.sheet_names
                    selected_sheet = st.selectbox("シート選択", sheet_names)
                    df = pd.read_excel(uploaded_file, sheet_name=selected_sheet)
                else:
                    df = pd.read_csv(uploaded_file)
                
                with st.expander("データ確認", expanded=False):
                    st.dataframe(df.head(5), height=150)
                
                # --- Multi-Series Selection ---
                st.subheader("プロットデータの選択")
                
                cols = list(df.columns)
                colors_preset = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
                
                for i in range(1, 6):
                    # 有効化チェックボックス (Series 1は常にTrue)
                    is_enabled = True
                    if i > 1:
                        is_enabled = st.checkbox(f"データセット #{i} を追加", value=False, key=f"ds_en_{i}")
                    
                    if is_enabled:
                        # ★ここを変更: st.container() -> st.expander()
                        # 最初に追加したときやSeries1は開いておく
                        with st.expander(f"Series {i} 設定", expanded=True):
                            c_vars1, c_vars2 = st.columns(2)
                            x_c = c_vars1.selectbox(f"X軸 #{i}", cols, index=0, key=f"xc_{i}")
                            y_c = c_vars2.selectbox(f"Y軸 #{i}", cols, index=1 if len(cols)>1 else 0, key=f"yc_{i}")
                            
                            c_sty1, c_sty2 = st.columns(2)
                            label_def = y_c
                            s_label = c_sty1.text_input(f"凡例名 #{i}", label_def, key=f"slb_{i}")
                            s_color = c_sty2.color_picker(f"色 #{i}", colors_preset[i-1], key=f"scl_{i}")
                            
                            s_marker = st.selectbox(f"マーカー #{i}", ["o (丸)", "s (四角)", "^ (三角)", "D (ひし形)", "x (バツ)", "None (なし)"], key=f"smk_{i}")
                            
                            all_series_config.append({
                                "id": i, "x_col": x_c, "y_col": y_c, 
                                "label": s_label, "color": s_color, 
                                "marker": s_marker.split()[0]
                            })

            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.info("データ待機中...")

    # ---------------------------------------------------------
    # Column 2: Settings
    # ---------------------------------------------------------
    with col_settings:
        st.header("2. Settings")
        
        if df is None or not all_series_config:
            st.warning("先にデータを選択してください")
            st.stop()
            
        # --- A. Appearance ---
        with st.expander("表示設定 (Style)", expanded=True):
            plot_title = st.text_input("グラフタイトル", "Graph Title")
            c_lbl1, c_lbl2 = st.columns(2)
            x_label = c_lbl1.text_input("X軸ラベル", all_series_config[0]['x_col'])
            y_label = c_lbl2.text_input("Y軸ラベル", "Value")
            
            c_leg1, c_leg2 = st.columns(2)
            show_legend = c_leg1.checkbox("凡例を表示", True)
            
            loc_options = {
                "best": "自動 (best)", 
                "upper right": "右上", "upper left": "左上", 
                "lower right": "右下", "lower left": "左下",
                "center right": "右中央", "center left": "左中央",
                "upper center": "上中央", "lower center": "下中央"
            }
            leg_loc_key = c_leg2.selectbox("凡例の位置", list(loc_options.keys()), format_func=lambda x: loc_options[x], index=0)

            st.caption("目盛り設定")
            c_tick1, c_tick2 = st.columns(2)
            tick_direction = c_tick1.radio("目盛り向き", ["in (内側)", "out (外側)"], index=0)
            minor_ticks = c_tick2.checkbox("中目盛りを表示", True)
            
            c_siz1, c_siz2 = st.columns(2)
            fig_width = c_siz1.number_input("幅", 2.0, 20.0, 6.0, step=0.5)
            fig_height = c_siz2.number_input("高さ", 2.0, 20.0, 4.5, step=0.5)
            base_fontsize = st.number_input("フォントサイズ", 8, 24, 14)

        # --- B. Axis Scale ---
        with st.expander("軸スケール (Scale)", expanded=False):
            st.write("軸の数値表記設定")
            c_sca1, c_sca2 = st.columns(2)
            x_log = c_sca1.checkbox("X: Log", False)
            y_log = c_sca2.checkbox("Y: Log", False)
            
            scale_mode_label = st.radio(
                "数値のスケーリング方法",
                ("SI接頭辞 (自動スケール)", "指数表記 (目盛りのみ変更)"),
                index=0
            )
            use_si_prefix = "SI接頭辞" in scale_mode_label

        # --- C. Fitting (Multi) ---
        with st.expander("近似直線・曲線 (Fitting)", expanded=True):
            st.caption("最大5本まで追加可能")
            
            # --- R2表示の有無 ---
            show_r2_legend = st.checkbox("近似線の凡例にR²値を表示する", value=True)
            # ---------------------------

            fit_configs = []
            
            try:
                all_x_vals = []
                for s in all_series_config:
                    all_x_vals.extend(pd.to_numeric(df[s['x_col']], errors='coerce').dropna().values)
                
                if not all_x_vals:
                    x_min_val, x_max_val = 0.0, 10.0
                else:
                    x_min_val = float(min(all_x_vals))
                    x_max_val = float(max(all_x_vals))
            except:
                x_min_val, x_max_val = 0.0, 10.0

            for i in range(1, 6):
                if st.checkbox(f"近似 #{i} を追加", value=False, key=f"fit_en_{i}"):
                    # 近似設定もついでに折り畳めるようにExpander化
                    with st.expander(f"Fit #{i} 設定", expanded=True):
                        target_series_idx = 0
                        if len(all_series_config) > 1:
                            series_options = [f"Series {s['id']}: {s['label']}" for s in all_series_config]
                            target_series_label = st.selectbox(f"対象データ #{i}", series_options, key=f"fts_{i}")
                            target_series_idx = series_options.index(target_series_label)
                        
                        c_f1, c_f2 = st.columns(2)
                        f_type = c_f1.selectbox(f"モデル #{i}", ["Linear", "Poly", "Exp"], key=f"ft_{i}")
                        f_deg = 1
                        if "Poly" in f_type:
                            f_deg = c_f2.slider(f"次数 #{i}", 1, 6, 2, key=f"dg_{i}")
                        
                        # 計算に使う範囲
                        range_vals = st.slider(
                            f"適用範囲 (計算用データ範囲) #{i}", 
                            min_value=x_min_val, 
                            max_value=x_max_val, 
                            value=(x_min_val, x_max_val),
                            key=f"rng_{i}"
                        )
                        
                        c_col, c_styl = st.columns(2)
                        f_color = c_col.color_picker(f"色 #{i}", value="#ff7f0e", key=f"fcl_{i}")
                        f_style = c_styl.selectbox(f"線種 #{i}", ["-- (破線)", "- (実線)", ": (点線)", "-. (一点鎖線)"], key=f"fst_{i}")
                        
                        c_lbl, c_ext = st.columns([2, 1.3])
                        f_label = c_lbl.text_input(f"凡例名 #{i}", f"Fit {i}", key=f"flb_{i}")
                        extend_fit = c_ext.checkbox("全範囲に延長", value=True, key=f"ext_{i}", help="チェックを入れると、近似線をグラフの端から端まで描画します（外挿）。")
                        
                        fit_configs.append({
                            "type": f_type, "deg": f_deg, "range": range_vals, 
                            "label": f_label, "style": f_style.split()[0], "color": f_color,
                            "target_idx": target_series_idx,
                            "extend": extend_fit
                        })

    # ---------------------------------------------------------
    # Column 3: Result
    # ---------------------------------------------------------
    with col_result:
        st.header("3. Result")
        
        max_x_abs = 0
        max_y_abs = 0
        
        # グローバルなX範囲を取得
        global_min_x = float('inf')
        global_max_x = float('-inf')

        plot_data_list = []
        for s_cfg in all_series_config:
            d_sub = df[[s_cfg['x_col'], s_cfg['y_col']]].apply(pd.to_numeric, errors='coerce').dropna()
            d_sub = d_sub.sort_values(by=s_cfg['x_col'])
            
            vx = d_sub[s_cfg['x_col']].values
            vy = d_sub[s_cfg['y_col']].values
            
            if len(vx) > 0:
                max_x_abs = max(max_x_abs, np.max(np.abs(vx)))
                max_y_abs = max(max_y_abs, np.max(np.abs(vy)))
                global_min_x = min(global_min_x, np.min(vx))
                global_max_x = max(global_max_x, np.max(vx))
            
            plot_data_list.append({"x": vx, "y": vy, "cfg": s_cfg})

        if global_min_x == float('inf'):
            global_min_x, global_max_x = 0, 10

        # --- スケールファクターの決定 ---
        scale_x, suffix_x = 1.0, ""
        scale_y, suffix_y = 1.0, ""

        if use_si_prefix:
            if not x_log:
                scale_x, suffix_x = get_optimal_scale(max_x_abs, use_si=True)
            if not y_log:
                scale_y, suffix_y = get_optimal_scale(max_y_abs, use_si=True)
            
            if suffix_x: 
                st.info(f"ℹ️ X軸: 単位 **{suffix_x.strip()}** に合わせて数値をスケーリングしました。必要に応じてラベルを修正してください。")
            if suffix_y: 
                st.info(f"ℹ️ Y軸: 単位 **{suffix_y.strip()}** に合わせて数値をスケーリングしました。必要に応じてラベルを修正してください。")
        else:
            st.info("ℹ️ 指数表記モード: 目盛りの数値を $a \\times 10^b$ 形式で表示します")

        # --- 表示範囲の計算 (重要) ---
        view_y_min = float('inf')
        view_y_max = float('-inf')
        has_data = False

        for pdata in plot_data_list:
            py = pdata['y'] * scale_y
            if len(py) > 0:
                view_y_min = min(view_y_min, np.min(py))
                view_y_max = max(view_y_max, np.max(py))
                has_data = True
        
        # --- Plot ---
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
        # 1. Plot Series
        for pdata in plot_data_list:
            px = pdata['x'] * scale_x
            py = pdata['y'] * scale_y
            cfg = pdata['cfg']
            
            marker = cfg['marker']
            if marker == "None": marker = None
            
            # edgecolors='none' で白枠なし
            ax.scatter(px, py, label=cfg['label'], color=cfg['color'], marker=marker, 
                       s=50, edgecolors='none', zorder=3)
            
        # 2. Plot Fits
        fit_results_md = ""
        for idx, fc in enumerate(fit_configs):
            target_idx = fc['target_idx']
            if target_idx < len(plot_data_list):
                pdata = plot_data_list[target_idx]
                raw_x = pdata['x']
                raw_y = pdata['y']
                
                # 計算用範囲でフィルタリング
                mask = (raw_x >= fc['range'][0]) & (raw_x <= fc['range'][1])
                
                if np.sum(mask) >= 2:
                    x_f_raw = raw_x[mask]
                    y_f_raw = raw_y[mask]
                    
                    try:
                        popt, eq_str, r2 = None, "", np.nan
                        
                        if "Linear" in fc['type']:
                            popt, _ = curve_fit(model_linear, x_f_raw, y_f_raw)
                            func = model_linear
                            # 数式整形 (LaTeX, 10^n表記)
                            a_str = fmt_latex_num(popt[0])
                            b_str = fmt_latex_num(abs(popt[1]))
                            sign = "+" if popt[1] >= 0 else "-"
                            eq_str = fr"y = {a_str}x {sign} {b_str}"

                        elif "Poly" in fc['type']:
                            popt = np.polyfit(x_f_raw, y_f_raw, fc['deg'])
                            func = lambda x, *p: np.polyval(p, x)
                            eq_str = f"Poly(deg={fc['deg']})"

                        elif "Exp" in fc['type']:
                            popt, _ = curve_fit(model_exp, x_f_raw, y_f_raw, p0=[1, 0.1, 0], maxfev=5000)
                            func = model_exp
                            eq_str = "Exp"

                        y_pred_raw = func(x_f_raw, *popt)
                        r2 = calc_r2(y_f_raw, y_pred_raw)
                        
                        # ★ 線の描画範囲決定
                        if fc['extend']:
                            x_line_raw = np.linspace(global_min_x, global_max_x, 200)
                        else:
                            x_line_raw = np.linspace(min(x_f_raw), max(x_f_raw), 100)
                            
                        y_line_raw = func(x_line_raw, *popt)
                        
                        # ★ 凡例テキストの生成
                        if show_legend:
                            if show_r2_legend:
                                label_txt = f"{fc['label']}: ${eq_str}$ ($R^2={r2:.3f}$)"
                            else:
                                label_txt = f"{fc['label']}: ${eq_str}$"
                        else:
                            label_txt = None
                        
                        ax.plot(x_line_raw * scale_x, y_line_raw * scale_y, 
                               color=fc['color'], linestyle=fc['style'], linewidth=2, 
                               label=label_txt, zorder=4)
                        
                        fit_results_md += f"- **{fc['label']}** (Series {all_series_config[target_idx]['label']}): ${eq_str}$, $R^2={r2:.4f}$\n"
                        
                    except Exception as e:
                        fit_results_md += f"- **{fc['label']}**: Failed ({e})\n"

        # Styling
        ax.set_title(plot_title, fontsize=base_fontsize+2, fontweight='bold')
        
        ax.set_xlabel(x_label, fontsize=base_fontsize)
        ax.set_ylabel(y_label, fontsize=base_fontsize)
        
        ax.tick_params(labelsize=base_fontsize-2)
        t_dir = 'in' if 'in' in tick_direction else 'out'
        ax.tick_params(direction=t_dir, which='both', top=True, right=True)
        
        if minor_ticks:
            ax.minorticks_on()
            ax.tick_params(which='minor', direction=t_dir, top=True, right=True)
        
        # Log設定 または Scientific Ticks設定
        if x_log: 
            ax.set_xscale('log')
        elif not use_si_prefix:
            if max_x_abs > 0 and (max_x_abs < 0.01 or max_x_abs >= 1000):
                 ax.xaxis.set_major_formatter(FuncFormatter(force_sci_format_func))

        if y_log: 
            ax.set_yscale('log')
        elif not use_si_prefix:
            ax.yaxis.set_major_formatter(FuncFormatter(force_sci_format_func))

        # ★ 重要: Y軸の表示範囲を「データ点」に基づいて固定する
        if has_data and not y_log:
             y_margin = (view_y_max - view_y_min) * 0.05
             if y_margin == 0: y_margin = abs(view_y_max) * 0.1
             ax.set_ylim(view_y_min - y_margin, view_y_max + y_margin)
            
        ax.xaxis.get_offset_text().set_visible(False)
        ax.yaxis.get_offset_text().set_visible(False)

        if show_legend:
            ax.legend(fontsize=base_fontsize-2, frameon=True, fancybox=False, edgecolor='black', loc=leg_loc_key)
        
        try:
            plt.tight_layout()
        except Exception:
            pass

        st.pyplot(fig)
        
        if fit_results_md:
            st.info(fit_results_md)
            
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=300, bbox_inches='tight')
        st.download_button("Download PNG", buf.getvalue(), "plot.png", "image/png", use_container_width=True)

if __name__ == "__main__":
    main()