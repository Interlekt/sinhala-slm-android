package com.interlekt.slmengine

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.interlekt.slmengine.ui.theme.SinhalaQAEngineTheme

class MainActivity : ComponentActivity() {

    private val vm: InferenceViewModel by viewModels()

    private val permLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { boot() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // A 50-question batch runs ~45 minutes. If the screen sleeps, Android
        // may freeze or kill the process and the run is lost.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)

        setContent {
            SinhalaQAEngineTheme {
                Surface(Modifier.fillMaxSize()) { Screen(vm) }
            }
        }
        checkPerms()
    }

    private fun checkPerms() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                startActivity(android.content.Intent(
                    android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    android.net.Uri.parse("package:$packageName")))
            }
            boot()
        } else {
            val ok = ContextCompat.checkSelfPermission(
                this, Manifest.permission.READ_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED
            if (!ok) permLauncher.launch(arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE))
            else boot()
        }
    }

    private fun boot() {
        vm.discoverModels()
        vm.loadRag()
    }
}

@Composable
fun Screen(vm: InferenceViewModel) {
    val st by vm.state.collectAsState()
    val rag by vm.ragState.collectAsState()
    val exp by vm.experiment.collectAsState()
    val eval by vm.evalState.collectAsState()
    val models by vm.models.collectAsState()
    val current by vm.currentModel.collectAsState()

    var input by remember { mutableStateOf("") }
    var showSources by remember { mutableStateOf(false) }
    var showExperiment by remember { mutableStateOf(true) }
    var modelMenu by remember { mutableStateOf(false) }
    val scroll = rememberScrollState()

    LaunchedEffect(st.output) { scroll.animateScrollTo(scroll.maxValue) }

    val busy = eval.running || st.isLoading

    Column(Modifier.fillMaxSize().padding(12.dp)) {

        // ── status ───────────────────────────────────────────────────────────
        Text(st.statusMsg, style = MaterialTheme.typography.labelSmall,
            color = if (st.modelLoaded) MaterialTheme.colorScheme.primary
            else MaterialTheme.colorScheme.error)
        Text(rag.ragStatus, style = MaterialTheme.typography.labelSmall,
            color = if (rag.ragReady) MaterialTheme.colorScheme.primary
            else MaterialTheme.colorScheme.error)

        Spacer(Modifier.height(6.dp))

        // ── generator picker ─────────────────────────────────────────────────
        Box {
            OutlinedButton(onClick = { modelMenu = true }, enabled = !busy,
                modifier = Modifier.fillMaxWidth()) {
                Text(current?.let { "${it.quant}  (${it.sizeMb} MB)" }
                    ?: "Select generator (${models.size} found)")
            }
            DropdownMenu(expanded = modelMenu, onDismissRequest = { modelMenu = false }) {
                if (models.isEmpty()) {
                    DropdownMenuItem(text = { Text("none found") }, onClick = {})
                }
                models.forEach { m ->
                    DropdownMenuItem(
                        text = { Text("${m.quant}   ${m.sizeMb} MB\n${m.name}",
                            style = MaterialTheme.typography.labelSmall) },
                        onClick = { modelMenu = false; vm.loadSlm(m) })
                }
            }
        }

        Spacer(Modifier.height(6.dp))

        // ── experiment panel ─────────────────────────────────────────────────
        Text(if (showExperiment) "▼ Experiment" else "▶ Experiment",
            style = MaterialTheme.typography.labelMedium,
            color = MaterialTheme.colorScheme.primary,
            modifier = Modifier.clickable { showExperiment = !showExperiment })

        if (showExperiment) {
            Surface(shape = RoundedCornerShape(8.dp),
                color = MaterialTheme.colorScheme.surfaceVariant,
                modifier = Modifier.fillMaxWidth()) {
                Column(Modifier.padding(8.dp)) {

                    Toggle("RAG", exp.ragEnabled, !busy) { vm.setRag(it) }
                    Toggle("Boundary gate (abstention)", exp.gateEnabled,
                        !busy && exp.ragEnabled) { vm.setGate(it) }
                    Toggle("Selective context", exp.compressEnabled,
                        !busy && exp.ragEnabled) { vm.setCompress(it) }
                    Toggle("Retrieve only (no generation)", exp.retrieveOnly,
                        !busy && exp.ragEnabled) { vm.setRetrieveOnly(it) }
                    // No boundary-mode toggle: the gate is the logistic-
                    // regression rule from knowledge_boundary.py,
                    // unconditionally. The legacy and calibrated rules are
                    // still computed into every record's sig_* fields and
                    // evaluable offline via sweep_boundary.py.

                    Spacer(Modifier.height(4.dp))
                    Text("Token budget", style = MaterialTheme.typography.labelSmall)
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        listOf(256, 384, 512).forEach { n ->
                            FilterChip(selected = exp.tokenBudget == n,
                                onClick = { vm.setTokenBudget(n) },
                                enabled = !busy && exp.compressEnabled,
                                label = { Text("$n", style = MaterialTheme.typography.labelSmall) })
                        }
                    }

                    Text("Cooldown between questions",
                        style = MaterialTheme.typography.labelSmall,
                        modifier = Modifier.padding(top = 4.dp))
                    Row(horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                        listOf(0L, 20_000L, 45_000L).forEach { ms ->
                            FilterChip(selected = exp.cooldownMs == ms,
                                onClick = { vm.setCooldown(ms) }, enabled = !busy,
                                label = { Text(if (ms == 0L) "off" else "${ms / 1000}s",
                                    style = MaterialTheme.typography.labelSmall) })
                        }
                    }

                    Spacer(Modifier.height(6.dp))
                    Text("run tag:  ${vm.runTag()}",
                        style = MaterialTheme.typography.labelSmall,
                        fontFamily = FontFamily.Monospace,
                        fontWeight = FontWeight.Bold)
                }
            }
        }

        Spacer(Modifier.height(6.dp))

        // ── batch ────────────────────────────────────────────────────────────
        if (eval.running) {
            LinearProgressIndicator(
                progress = { if (eval.total > 0) eval.done.toFloat() / eval.total else 0f },
                modifier = Modifier.fillMaxWidth())
            Text("${eval.done}/${eval.total}   ${eval.currentId}",
                style = MaterialTheme.typography.labelSmall)
            OutlinedButton(onClick = { vm.cancelBatch() }, Modifier.fillMaxWidth()) {
                Text("Cancel batch")
            }
        } else {
            Button(onClick = { vm.runBatch() }, Modifier.fillMaxWidth(),
                enabled = st.modelLoaded && !busy) { Text("Run batch") }
        }
        if (eval.message.isNotEmpty()) {
            Text(eval.message, style = MaterialTheme.typography.labelSmall)
        }

        Spacer(Modifier.height(6.dp))

        // ── metrics ──────────────────────────────────────────────────────────
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
            Chip("TTFT ${st.ttftMs}ms", Modifier.weight(1f))
            Chip("${st.msPerToken.toInt()} ms/tok", Modifier.weight(1f))
            Chip("${st.promptTokens} tok", Modifier.weight(1f))
            Chip("${st.ramMB} MB", Modifier.weight(1f))
        }
        if (exp.ragEnabled && rag.ragReady) {
            Spacer(Modifier.height(3.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(4.dp)) {
                Chip("embed ${rag.embedMs}ms", Modifier.weight(1f))
                Chip("retrieve ${rag.retrieveMs}ms", Modifier.weight(1f))
                Chip("prompt ${rag.promptChars}ch", Modifier.weight(1f))
            }
        }

        // ── boundary verdict ─────────────────────────────────────────────────
        rag.boundary?.let { b ->
            Spacer(Modifier.height(6.dp))
            Surface(Modifier.fillMaxWidth(), shape = RoundedCornerShape(8.dp),
                color = if (b.isOutOfSyllabus) Color(0xFFFFE0E0) else Color(0xFFE0F5E0)) {
                Column(Modifier.padding(8.dp)) {
                    Text(if (b.isOutOfSyllabus)
                        "⛔ ${b.label} — model not invoked" else "✓ ${b.label}",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold, color = Color(0xFF202020))
                    if (b.probability >= 0.0) {
                        Text("P(within syllabus) = ${"%.3f".format(b.probability)}",
                            style = MaterialTheme.typography.labelSmall, color = Color(0xFF404040))
                    }
                    Text(b.reason, style = MaterialTheme.typography.labelSmall,
                        color = Color(0xFF505050))
                }
            }
        }

        Spacer(Modifier.height(6.dp))

        // ── output ───────────────────────────────────────────────────────────
        Box(Modifier.weight(1f).fillMaxWidth()
            .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(8.dp))
            .padding(10.dp)) {
            Text(st.output.ifEmpty { "Output appears here" },
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.verticalScroll(scroll))
            if (st.isLoading) {
                CircularProgressIndicator(Modifier.size(18.dp).align(Alignment.TopEnd),
                    strokeWidth = 2.dp)
            }
        }

        // ── sources ──────────────────────────────────────────────────────────
        if (rag.sources.isNotEmpty()) {
            Text(if (showSources) "▼ Sources (${rag.sources.size})"
            else "▶ Sources (${rag.sources.size})",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.clickable { showSources = !showSources })
            if (showSources) {
                Column(Modifier.fillMaxWidth().heightIn(max = 180.dp)
                    .verticalScroll(rememberScrollState())) {
                    rag.sources.forEach { SourceRow(it) }
                }
            }
        }

        Spacer(Modifier.height(6.dp))

        OutlinedTextField(value = input, onValueChange = { input = it },
            modifier = Modifier.fillMaxWidth(), maxLines = 3, enabled = !busy,
            label = { Text("ප්‍රශ්නය") })

        Spacer(Modifier.height(4.dp))

        Button(onClick = { vm.ask(input) }, Modifier.fillMaxWidth(),
            enabled = st.modelLoaded && !busy) {
            Text(when {
                st.isLoading -> "Generating..."
                exp.ragEnabled && exp.retrieveOnly -> "Retrieve"
                exp.ragEnabled -> "Ask (RAG)"
                else -> "Generate"
            })
        }
    }
}

@Composable
private fun Toggle(label: String, checked: Boolean, enabled: Boolean, onChange: (Boolean) -> Unit) {
    Row(verticalAlignment = Alignment.CenterVertically) {
        Switch(checked = checked, onCheckedChange = onChange, enabled = enabled)
        Text(label, Modifier.padding(start = 6.dp),
            style = MaterialTheme.typography.labelSmall)
    }
}

@Composable
private fun SourceRow(r: Retrieved) {
    Surface(Modifier.fillMaxWidth().padding(vertical = 2.dp),
        shape = RoundedCornerShape(6.dp), color = MaterialTheme.colorScheme.surfaceVariant) {
        Column(Modifier.padding(6.dp)) {
            Text("#${r.rank}  fused=${"%.5f".format(r.score)}  cos=${"%.4f".format(r.cosine)}",
                style = MaterialTheme.typography.labelSmall, fontWeight = FontWeight.Bold)
            Text(listOf(r.chunk.grade, r.chunk.chapter, "p.${r.chunk.page}")
                .filter { it.isNotBlank() && it != "p." }.joinToString(" / "),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant)
            Text(r.chunk.text.take(110).replace("\n", " ") + "…",
                style = MaterialTheme.typography.labelSmall)
        }
    }
}

@Composable
fun Chip(label: String, modifier: Modifier = Modifier) {
    Surface(modifier, shape = RoundedCornerShape(20.dp),
        color = MaterialTheme.colorScheme.secondaryContainer) {
        Text(label, Modifier.padding(horizontal = 6.dp, vertical = 3.dp),
            style = MaterialTheme.typography.labelSmall)
    }
}
