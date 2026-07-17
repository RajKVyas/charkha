// Ghidra headless post-script (Java) — append each function's decompiled C to a JSONL.
// Invoked by scripts/build_re_dataset.py via analyzeHeadless -postScript with one arg
// (the output JSONL path). Runs ONCE PER IMPORTED PROGRAM, so a single headless session
// over a whole directory of .o files amortizes JVM/analysis startup across all of them.
// Each line: {"program":"<.o name>","func":"<sym>","pseudo":"<decompiled C>"} keyed by
// program so the Python side can match back to the right (source,opt) rung. Java scripts
// run natively under headless Ghidra (no PyGhidra needed); JSON is escaped by hand to
// avoid any external dependency.
//@category CHARKHA
import java.io.FileWriter;
import ghidra.app.script.GhidraScript;
import ghidra.app.decompiler.DecompInterface;
import ghidra.app.decompiler.DecompileResults;
import ghidra.program.model.listing.Function;
import ghidra.program.model.listing.FunctionManager;
import ghidra.util.task.ConsoleTaskMonitor;

public class DecompileToJson extends GhidraScript {
    private static String esc(String s) {
        StringBuilder b = new StringBuilder();
        for (int i = 0; i < s.length(); i++) {
            char c = s.charAt(i);
            switch (c) {
                case '"':  b.append("\\\""); break;
                case '\\': b.append("\\\\"); break;
                case '\n': b.append("\\n");  break;
                case '\r': b.append("\\r");  break;
                case '\t': b.append("\\t");  break;
                default:
                    if (c < 0x20) b.append(String.format("\\u%04x", (int) c));
                    else b.append(c);
            }
        }
        return b.toString();
    }

    @Override
    public void run() throws Exception {
        String[] args = getScriptArgs();
        String outPath = args.length > 0 ? args[0] : "pseudo.jsonl";
        String prog = currentProgram.getName();
        DecompInterface decomp = new DecompInterface();
        decomp.openProgram(currentProgram);
        ConsoleTaskMonitor mon = new ConsoleTaskMonitor();
        FunctionManager fm = currentProgram.getFunctionManager();
        int n = 0;
        // append mode: one headless session imports many .o files, this runs per program
        try (FileWriter w = new FileWriter(outPath, true)) {
            for (Function f : fm.getFunctions(true)) {
                try {
                    DecompileResults r = decomp.decompileFunction(f, 60, mon);
                    if (r != null && r.decompileCompleted()) {
                        String c = r.getDecompiledFunction().getC();
                        w.write("{\"program\":\"" + esc(prog) + "\",\"func\":\""
                                + esc(f.getName()) + "\",\"pseudo\":\"" + esc(c) + "\"}\n");
                        n++;
                    }
                } catch (Exception e) {
                    // skip undecompilable functions
                }
            }
        }
        println("[DecompileToJson] " + prog + ": wrote " + n + " functions -> " + outPath);
    }
}
