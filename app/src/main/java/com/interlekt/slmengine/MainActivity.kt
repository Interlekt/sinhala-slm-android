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
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import com.interlekt.slmengine.ui.theme.SinhalaQAEngineTheme

class MainActivity : ComponentActivity() {

    private val vm: InferenceViewModel by viewModels()

    // Runtime permission launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestMultiplePermissions()
    ) { permissions ->
        // After user responds to permission dialog — try loading
        loadModel()
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContent {
            SinhalaQAEngineTheme {
                Surface(modifier = Modifier.fillMaxSize()) {
                    InferenceScreen(vm)
                }
            }
        }
        checkPermissionsAndLoad()
    }

    private fun checkPermissionsAndLoad() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            // Android 11+ — request MANAGE_EXTERNAL_STORAGE
            if (!Environment.isExternalStorageManager()) {
                val intent = android.content.Intent(
                    android.provider.Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION,
                    android.net.Uri.parse("package:$packageName")
                )
                startActivity(intent)
                // Load after user comes back
                loadModel()
            } else {
                loadModel()
            }
        } else {
            // Android 9/10
            val readGranted = ContextCompat.checkSelfPermission(
                this, Manifest.permission.READ_EXTERNAL_STORAGE
            ) == PackageManager.PERMISSION_GRANTED

            if (!readGranted) {
                requestPermissionLauncher.launch(
                    arrayOf(Manifest.permission.READ_EXTERNAL_STORAGE)
                )
            } else {
                loadModel()
            }
        }
    }

    private fun loadModel() {
        val modelPath = "/sdcard/Download/qwen2.5-1.5b-instruct-q4_k_m.gguf"
        vm.loadModel(modelPath)
    }
}

@Composable
fun InferenceScreen(vm: InferenceViewModel) {
    val state by vm.state.collectAsState()
    var input by remember { mutableStateOf("") }
    val scrollState = rememberScrollState()

    LaunchedEffect(state.output) {
        scrollState.animateScrollTo(scrollState.maxValue)
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(16.dp)
    ) {
        Text(
            text = state.statusMsg,
            style = MaterialTheme.typography.labelSmall,
            color = if (state.modelLoaded)
                MaterialTheme.colorScheme.primary
            else MaterialTheme.colorScheme.error
        )

        Spacer(Modifier.height(8.dp))

        Row(
            modifier = Modifier.fillMaxWidth(),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            MetricChip("${state.msPerToken.toInt()} ms/tok", Modifier.weight(1f))
            MetricChip("${state.ramMB} MB RAM", Modifier.weight(1f))
            MetricChip("${state.cpuPct.toInt()}% CPU", Modifier.weight(1f))
        }

        Spacer(Modifier.height(12.dp))

        Box(
            modifier = Modifier
                .weight(1f)
                .fillMaxWidth()
                .border(1.dp,
                    MaterialTheme.colorScheme.outlineVariant,
                    RoundedCornerShape(8.dp))
                .padding(12.dp)
        ) {
            Text(
                text = state.output.ifEmpty { "Output streams here..." },
                style = MaterialTheme.typography.bodySmall,
                modifier = Modifier.verticalScroll(scrollState)
            )
            if (state.isLoading) {
                CircularProgressIndicator(
                    modifier = Modifier
                        .size(20.dp)
                        .align(Alignment.TopEnd),
                    strokeWidth = 2.dp
                )
            }
        }

        Spacer(Modifier.height(12.dp))

        OutlinedTextField(
            value = input,
            onValueChange = { input = it },
            modifier = Modifier.fillMaxWidth(),
            label = { Text("Enter your prompt") },
            maxLines = 3
        )

        Spacer(Modifier.height(8.dp))

        Button(
            onClick = { vm.generate(input) },
            modifier = Modifier.fillMaxWidth(),
            enabled = state.modelLoaded && !state.isLoading
        ) {
            Text(if (state.isLoading) "Generating..." else "Generate")
        }
    }
}

@Composable
fun MetricChip(label: String, modifier: Modifier = Modifier) {
    Surface(
        modifier = modifier,
        shape = RoundedCornerShape(20.dp),
        color = MaterialTheme.colorScheme.secondaryContainer
    ) {
        Text(
            text = label,
            modifier = Modifier.padding(horizontal = 8.dp, vertical = 4.dp),
            style = MaterialTheme.typography.labelSmall
        )
    }
}
