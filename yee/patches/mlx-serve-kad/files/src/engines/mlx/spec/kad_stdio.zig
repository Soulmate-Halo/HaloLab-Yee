//! std.Io file access and clock for the Zig KAD core (Zig 0.17 std.Io API).
//!
//! `Files.table()` is the `kad.Io` table of `kad.Kad.init`: stat / read of the policy file and an atomic
//! write of the statistics file (temporary file, then rename; the directory is created). `nowNs` is the
//! monotonic clock `kad.Kad.beginStep` takes.
const std = @import("std");
const kad = @import("kad.zig");

pub const Files = struct {
    io: std.Io,

    pub fn table(self: *Files) kad.Io {
        return .{ .ctx = self, .stat = &stat, .read = &read, .write = &write };
    }

    pub fn nowNs(self: *const Files) i128 {
        return std.Io.Clock.awake.now(self.io).toNanoseconds();
    }

    fn stat(ctx: ?*anyopaque, path: []const u8) ?kad.Stamp {
        const self: *Files = @ptrCast(@alignCast(ctx.?));
        const st = std.Io.Dir.cwd().statFile(self.io, path, .{}) catch return null;
        return .{ .mtime_ns = st.mtime.nanoseconds, .size = st.size };
    }

    fn read(ctx: ?*anyopaque, gpa: std.mem.Allocator, path: []const u8, limit: usize) ?[]u8 {
        const self: *Files = @ptrCast(@alignCast(ctx.?));
        return std.Io.Dir.cwd().readFileAlloc(self.io, path, gpa, .limited(limit)) catch null;
    }

    fn write(ctx: ?*anyopaque, path: []const u8, bytes: []const u8) bool {
        const self: *Files = @ptrCast(@alignCast(ctx.?));
        const cwd = std.Io.Dir.cwd();
        if (std.fs.path.dirname(path)) |dir| cwd.createDirPath(self.io, dir) catch return false;
        var buf: [4096]u8 = undefined;
        const tmp = std.fmt.bufPrint(&buf, "{s}.tmp", .{path}) catch return false;
        cwd.writeFile(self.io, .{ .sub_path = tmp, .data = bytes }) catch return false;
        std.Io.Dir.rename(cwd, tmp, cwd, path, self.io) catch {
            cwd.deleteFile(self.io, tmp) catch {};
            return false;
        };
        return true;
    }
};

test "Files: stat, read, atomic write, clock" {
    var tmp = std.testing.tmpDir(.{});
    defer tmp.cleanup();
    var files: Files = .{ .io = std.testing.io };
    const t = files.table();
    var pb: [512]u8 = undefined;
    const p = try std.fmt.bufPrint(&pb, ".zig-cache/tmp/{s}/sub/stats.json", .{&tmp.sub_path});
    try std.testing.expect(t.stat(t.ctx, p) == null);
    try std.testing.expect(t.write(t.ctx, p, "{\"a\": 1}"));
    try std.testing.expect(t.write(t.ctx, p, "{\"a\": 12}"));
    const st = t.stat(t.ctx, p).?;
    try std.testing.expectEqual(@as(u64, 9), st.size);
    const got = t.read(t.ctx, std.testing.allocator, p, 1024).?;
    defer std.testing.allocator.free(got);
    try std.testing.expectEqualStrings("{\"a\": 12}", got);
    try std.testing.expect(t.read(t.ctx, std.testing.allocator, p, 4) == null);
    const a = files.nowNs();
    const b = files.nowNs();
    try std.testing.expect(b >= a and a > 0);
}
