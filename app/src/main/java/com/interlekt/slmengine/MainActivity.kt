package com.interlekt.slmengine

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import android.os.Environment
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
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
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.interlekt.slmengine.ui.theme.SinhalaQAEngineTheme

class MainActivity : ComponentActivity() {

    private val vm: InferenceViewModel by viewModels()

    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { loadEverything() }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SinhalaQAEngineTheme {
                Surface(modifier = Modifier.fillMaxSize()) { InferenceScreen(vm) }
            }
        }
        checkPermissionsAndLoad()
    }

    private fun checkPermissionsAndLoad() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                startActivity(
                    android.content.Intent(
                        android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                        android.net.Uri.parse("package:$packageName")
                    )
                )
            }
            loadEverything()
        } else {
            val granted = ContextCompat.checkSelfPermission(
                this, Manifest.permission.READ_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED
            if (!granted) {
                requestPermissionLauncher.launch(arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE))
            } else loadEverything()
        }
    }

    /**
     * Generator first, then embedder + corpus. Loading the generator first means
     * the app is usable in no-RAG mode while the ~610 MB embedder streams in.
     */
    private fun loadEverything() {
        vm.loadModel("/sdcard/Download/sinllama.gguf")
        vm.loadRag(
            embedderPath = "/data/local/tmp/bge-m3.gguf",
            bundleDir = "/sdcard/Download/rag_bundle",
        )
    }
}

@Composable
fun InferenceScreen(vm: InferenceViewModel) {
    val state by vm.state.collectAsState()
    val rag by vm.ragState.collectAsState()
    var input by remember { mutableStateOf("") }
    var showSources by remember { mutableStateOf(false) }
    val scrollState = rememberScrollState()

    LaunchedEffect(state.output) { scrollState.animateScrollTo(scrollState.maxValue) }

    Column(modifier = Modifier.fillMaxSize().padding(16.dp)) {

        // ── Status ───────────────────────────────────────────────────────────
        Text(
            text = state.statusMsg,
            style = MaterialTheme.typography.labelSmall,
            color = if (state.modelLoaded) MaterialTheme.colorScheme.primary
            else MaterialTheme.colorScheme.error,
        )
        Text(
            text = rag.ragStatus,
            style = MaterialTheme.typography.labelSmall,
            color = if (rag.ragReady) MaterialTheme.colorScheme.primary
            else MaterialTheme.colorScheme.error,
        )

        Spacer(Modifier.height(8.dp))

        // ── Controls. Both toggles exist because the ablation needs them. ────
        Row(verticalAlignment = Alignment.CenterVertically) {
            Switch(
                checked = rag.useRag,
                onCheckedChange = { vm.setUseRag(it) },
                enabled = rag.ragReady,
            )
            Text("RAG", Modifier.padding(start = 4.dp),
                style = MaterialTheme.typography.labelSmall)

            Spacer(Modifier.width(12.dp))

            Switch(
                checked = rag.retrieveOnly,
                onCheckedChange = { vm.setRetrieveOnly(it) },
                enabled = rag.ragReady && rag.useRag,
            )
            Text("Retrieve only", Modifier.padding(start = 4.dp),
                style = MaterialTheme.typography.labelSmall)
        }

        Row(verticalAlignment = Alignment.CenterVertically) {
            Switch(
                checked = rag.mode == BoundaryMode.CALIBRATED,
                onCheckedChange = {
                    vm.setBoundaryMode(
                        if (it) BoundaryMode.CALIBRATED else BoundaryMode.DESKTOP_PARITY
                    )
                },
                enabled = rag.ragReady,
            )
            Text(
                if (rag.mode == BoundaryMode.CALIBRATED) "Boundary: calibrated"
                else "Boundary: desktop parity",
                Modifier.padding(start = 4.dp),
                style = MaterialTheme.typography.labelSmall,
            )
        }

        Spacer(Modifier.height(8.dp))

        // ── Metrics. TTFT and decode are separate; a single blended figure
        //    would hide which half a RAG prompt actually costs. ──────────────
        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
            MetricChip("TTFT ${state.ttftMs}ms", Modifier.weight(1f))
            MetricChip("${state.msPerToken.toInt()} ms/tok", Modifier.weight(1f))
            MetricChip("${state.ramMB} MB", Modifier.weight(1f))
        }
        if (rag.useRag && rag.ragReady) {
            Spacer(Modifier.height(4.dp))
            Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                MetricChip("embed ${rag.embedMs}ms", Modifier.weight(1f))
                MetricChip("retrieve ${rag.retrieveMs}ms", Modifier.weight(1f))
                MetricChip("prompt ${rag.promptChars}ch", Modifier.weight(1f))
            }
        }

        // ── Boundary verdict ─────────────────────────────────────────────────
        rag.boundary?.let { b ->
            Spacer(Modifier.height(8.dp))
            Surface(
                modifier = Modifier.fillMaxWidth(),
                shape = RoundedCornerShape(8.dp),
                color = if (b.isOutOfSyllabus) Color(0xFFFFE0E0) else Color(0xFFE0F5E0),
            ) {
                Column(Modifier.padding(10.dp)) {
                    Text(
                        text = if (b.isOutOfSyllabus)
                            "⛔ ${b.label} — answered without the model"
                        else "✓ ${b.label}",
                        style = MaterialTheme.typography.labelMedium,
                        fontWeight = FontWeight.Bold,
                        color = Color(0xFF202020),
                    )
                    Text(
                        text = b.reason,
                        style = MaterialTheme.typography.labelSmall,
                        color = Color(0xFF505050),
                    )
                }
            }
        }

        Spacer(Modifier.height(8.dp))

        // ── Output ───────────────────────────────────────────────────────────
        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .border(1.dp, MaterialTheme.colorScheme.outlineVariant, RoundedCornerShape(8.dp))
                .padding(12.dp)
        ) {
            Text(
                text = state.output.ifEmpty { "Output streams here..." },
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.verticalScroll(scrollState),
            )
            if (state.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier.size(20.dp).align(Alignment.TopEnd),
                    strokeWidth = 2.dp,
                )
            }
        }

        // ── Sources. Not decoration: this is how an annotator checks whether
        //    an answer was actually grounded in what was retrieved. ──────────
        if (rag.sources.isNotEmpty()) {
            Spacer(Modifier.height(8.dp))
            Text(
                text = if (showSources) "▼ Sources (${rag.sources.size})"
                else "▶ Sources (${rag.sources.size})",
                style = MaterialTheme.typography.labelMedium,
                color = MaterialTheme.colorScheme.primary,
                modifier = Modifier.clickable { showSources = !showSources },
            )
            if (showSources) {
                Column(
                    Modifier
                        .fillMaxWidth()
                        .heightIn(max = 200.dp)
                        .verticalScroll(rememberScrollState())
                ) {
                    rag.sources.forEach { r -> SourceRow(r) }
                }
            }
        }

        Spacer(Modifier.height(8.dp))

        OutlinedTextField(
            value = input,
            onValueChange = { input = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("ප්‍රශ්නය ඇතුළත් කරන්න") },
            maxLines = 3,
        )

        Spacer(Modifier.height(8.dp))

        Button(
            onClick = { vm.ask(input) },
            modifier = Modifier.fillMaxWidth(),
            enabled = state.modelLoaded && !state.isLoading,
        ) {
            Text(
                when {
                    state.isLoading -> "Generating..."
                    rag.useRag && rag.retrieveOnly -> "Retrieve"
                    rag.useRag -> "Ask (RAG)"
                    else -> "Generate"
                }
            )
        }

        val evalStatus by vm.evalStatus.collectAsState()
        if (evalStatus.isNotEmpty()) {
            Text(evalStatus, style = MaterialTheme.typography.labelSmall)
        }
        Button(onClick = { vm.runBatch("Q4_K_M") }, enabled = state.modelLoaded) {
            Text("Run batch")
        }
    }
}

@Composable
private fun SourceRow(r: Retrieved) {
    Surface(
        modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp),
        shape = RoundedCornerShape(6.dp),
        color = MaterialTheme.colorScheme.surfaceVariant,
    ) {
        Column(Modifier.padding(8.dp)) {
            Text(
                text = "#${r.rank}  ${"%.5f".format(r.score)}",
                style = MaterialTheme.typography.labelSmall,
                fontWeight = FontWeight.Bold,
            )
            Text(
                text = listOf(r.chunk.grade, r.chunk.chapter, "p.${r.chunk.page}")
                    .filter { it.isNotBlank() && it != "p." }
                    .joinToString(" / "),
                style = MaterialTheme.typography.labelSmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Text(
                text = r.chunk.text.take(120).replace("\n", " ") + "…",
                style = MaterialTheme.typography.labelSmall,
            )
        }
    }
}

@Composable
fun MetricChip(label: String, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(20.dp),
        color = MaterialTheme.colorScheme.secondaryContainer,
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            style = MaterialTheme.typography.labelSmall,
        )
    }
}
