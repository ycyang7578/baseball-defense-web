export const translations = {
  zh: {
    nav: { positioning: '最佳化站位', rankings: 'OAA 排名' },

    gameState: { runnersOnBase: '壘上跑者', outs: '出局數', out0: '0 出局', out1: '1 出局', out2: '2 出局' },

    searchSelect: { noMatches: '無符合' },

    panel: {
      batter: '打者',
      batterPlaceholder: '搜尋打者…',
      batterHint: '括號內為該年場內球數。內外野七人一起排：飛球交給外野、滾地球交給內野',
      gameState: '比賽狀況',
      park: '球場',
      parkIndependent: '球場（各組獨立）',
      parkGeneric: '— 通用 —',
      parkHint: '指定球場會把打到牆的球視為必失分，外野站位跟著調整（計算較久）',
      fielders: '野手',
      minOpp: '最低守備次數',
      fielderHint: '括號內為模型估計 OAA/100，非 Statcast 官方數值。指定野手時以該球員的守備參數微調站位',
      comboA: '組合 A',
      comboB: '組合 B',
      closeCompare: '✕ 關閉比較模式',
      compareMode: '⇔ 比較模式',
      calculating: '計算中…',
      calculate: '計算最佳站位',
      leagueAvg: '聯盟平均',
    },

    emptyState: {
      title: '選擇打者與壘況開始分析',
      body: '七名野手一起排：外野三人對付飛球、內野四人對付滾地球，以「預期失分」同一把尺衡量，加總就是這套站位替球隊省下的分數',
    },

    overlay: { computing: '七人站位計算中（約需一分鐘）…' },

    titleBar: {
      yearStand: '（{year}, {stand}打）',
      situation: '壘況',
      flyPlusLine: '飛球 {fly} 球＋平飛 {line} 球',
      ofTotal: '外野 {n} 球',
      plusGb: '＋滾地 {n} 球',
      plusPopup: '＋高飛 {n} 球（展示）',
      gbInsufficient: '　（滾地球樣本不足，只排外野三人）',
    },

    statsPanel: {
      leagueAvgPositions: '聯盟平均站位',
      optimizedPositions: '最佳化站位',
      catchRateOf: '外野接殺率',
      outRateIf: '內野出局率',
      expectedRuns: '預期失分',
      ofIfBreakdown: '　（外野 {of}＋內野 {if}）',
      leagueAvgShort: '聯盟平均',
      optimizedShort: '最佳化',
    },

    compareStats: {
      comboA: '組合 A',
      comboB: '組合 B',
      diff: 'A − B',
      runsSavedTotal: '多守下幾分（總）',
      runsSavedOf: '多守下幾分（外野）',
      runsSavedIf: '多守下幾分（內野）',
    },

    chart: {
      toggleOwner: '切換：責任歸屬色',
      toggleProb: '切換：接殺機率色',
      density: '落點密度',
      download: '↓ 下載圖',
      probRange: '機率範圍',
      ballType: '球種',
      ballTypes: { ground_ball: '滾地球', fly_ball: '飛球', line_drive: '平飛球', popup: '內野高飛' },
      legend: {
        optimizedPositions: '最佳化站位',
        catchOut: '接殺/出局',
        other: '其他',
        clickHint: '點星標高亮其責任球',
        wallBalls: '打牆球 ({n})',
      },
      countLine: {
        flyLine: '飛球 {fly}・平飛 {line}',
        ofTotal: '外野 {n} 球',
        gb: '滾地 {n}',
        popup: '・高飛 {n}',
      },
      tooltip: {
        lineDrive: '平飛球', flyBall: '飛球', ofBall: '外野球',
        catchProb: '接殺機率',
        owner: '　歸屬 {code}',
        popupLabel: '內野高飛',
        out: '出局', hitError: '安打/失誤',
        popupCatchProb: '接殺機率 ~99%（實證）',
        popupHint: '站哪都接得到，不參與站位優化',
        gbLabel: '滾地球',
        avgToOpt: 'P(out) 平均 {avg}% → 最佳化 {opt}%',
      },
    },

    rankings: {
      title: 'OAA 守備排名',
      subtitle: '模型 OAA 以守備難度計算（外野=飛球接殺、內野=滾地出局，基於賽季平均站位，非 Statcast 官方數值）；右側官方欄位為 Statcast OAA',
      teamFilter: '球隊篩選',
      allTeams: '全部球隊',
      minOppModel: '最低守備機會（模型）',
      loading: '載入中…',
      columns: {
        player: '球員', team: '球隊', position: '守位',
        modelOpp: '模型機會', modelOaa: '模型OAA', oaaPer100: 'OAA/100',
        officialOaa: '官方OAA', officialOpp: '官方機會', officialRate: '官方OAA/100',
      },
    },
  },

  en: {
    nav: { positioning: 'Positioning', rankings: 'OAA Rankings' },

    gameState: { runnersOnBase: 'Runners on base', outs: 'Outs', out0: '0 outs', out1: '1 out', out2: '2 outs' },

    searchSelect: { noMatches: 'No matches' },

    panel: {
      batter: 'Batter',
      batterPlaceholder: 'Search batter…',
      batterHint: 'Number in parentheses is balls in play that year. All seven fielders are optimized together: fly balls to the outfield, ground balls to the infield.',
      gameState: 'Game state',
      park: 'Ballpark',
      parkIndependent: 'Ballpark (per group)',
      parkGeneric: '— Generic —',
      parkHint: 'Picking a ballpark treats wall balls as guaranteed hits and adjusts outfield positioning accordingly (slower to compute)',
      fielders: 'Fielders',
      minOpp: 'Min. fielding chances',
      fielderHint: 'Number in parentheses is model-estimated OAA/100, not an official Statcast value. Picking a fielder fine-tunes positioning using that player\'s defensive parameters.',
      comboA: 'Combo A',
      comboB: 'Combo B',
      closeCompare: '✕ Exit compare mode',
      compareMode: '⇔ Compare mode',
      calculating: 'Calculating…',
      calculate: 'Calculate optimal positioning',
      leagueAvg: 'League average',
    },

    emptyState: {
      title: 'Pick a batter and game state to begin',
      body: 'All seven fielders are positioned together: three outfielders against fly balls, four infielders against ground balls, both measured on the same "expected runs" scale — the total is how many runs this positioning saves the defense.',
    },

    overlay: { computing: 'Optimizing all seven positions (about a minute)…' },

    titleBar: {
      yearStand: ' ({year}, bats {stand})',
      situation: 'Situation',
      flyPlusLine: '{fly} fly + {line} line drive',
      ofTotal: '{n} outfield balls',
      plusGb: ' + {n} ground balls',
      plusPopup: ' + {n} popups (shown)',
      gbInsufficient: ' (too few ground balls — outfield only)',
    },

    statsPanel: {
      leagueAvgPositions: 'League-average positions',
      optimizedPositions: 'Optimized positions',
      catchRateOf: 'Outfield catch rate',
      outRateIf: 'Infield out rate',
      expectedRuns: 'Expected runs',
      ofIfBreakdown: '  (OF {of} + IF {if})',
      leagueAvgShort: 'League avg',
      optimizedShort: 'Optimized',
    },

    compareStats: {
      comboA: 'Combo A',
      comboB: 'Combo B',
      diff: 'A − B',
      runsSavedTotal: 'Runs saved (total)',
      runsSavedOf: 'Runs saved (outfield)',
      runsSavedIf: 'Runs saved (infield)',
    },

    chart: {
      toggleOwner: 'Switch: ownership color',
      toggleProb: 'Switch: catch-probability color',
      density: 'Landing density',
      download: '↓ Download',
      probRange: 'Probability range',
      ballType: 'Ball type',
      ballTypes: { ground_ball: 'Ground ball', fly_ball: 'Fly ball', line_drive: 'Line drive', popup: 'Infield popup' },
      legend: {
        optimizedPositions: 'Optimized position',
        catchOut: 'Catch/out %',
        other: 'Other',
        clickHint: 'Click a star to highlight its balls',
        wallBalls: 'Wall balls ({n})',
      },
      countLine: {
        flyLine: 'Fly {fly} · Line {line}',
        ofTotal: '{n} outfield balls',
        gb: 'GB {n}',
        popup: ' · Popup {n}',
      },
      tooltip: {
        lineDrive: 'Line drive', flyBall: 'Fly ball', ofBall: 'Outfield ball',
        catchProb: 'Catch prob.',
        owner: '  owner {code}',
        popupLabel: 'Infield popup',
        out: 'out', hitError: 'hit/error',
        popupCatchProb: '~99% catch prob. (empirical)',
        popupHint: 'Catchable from anywhere — not part of optimization',
        gbLabel: 'Ground ball',
        avgToOpt: 'P(out) avg {avg}% → optimized {opt}%',
      },
    },

    rankings: {
      title: 'OAA Rankings',
      subtitle: 'Model OAA is computed from fielding difficulty (outfield = fly-ball catch, infield = ground-ball out) based on season-average positioning, not an official Statcast value; the right-hand columns are official Statcast OAA.',
      teamFilter: 'Team filter',
      allTeams: 'All teams',
      minOppModel: 'Min. fielding chances (model)',
      loading: 'Loading…',
      columns: {
        player: 'Player', team: 'Team', position: 'Pos',
        modelOpp: 'Model Opp', modelOaa: 'Model OAA', oaaPer100: 'OAA/100',
        officialOaa: 'Official OAA', officialOpp: 'Official Opp', officialRate: 'Official OAA/100',
      },
    },
  },
}
